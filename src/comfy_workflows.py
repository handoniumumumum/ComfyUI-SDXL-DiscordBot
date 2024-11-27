import configparser
import os

import PIL
import discord
import asyncio

from PIL import Image

from src.defaults import UPSCALE_DEFAULTS, MAX_RETRIES
from src.image_gen.ImageWorkflow import *
from src.image_gen.sd_workflows import *
from src.util import get_loras_from_prompt

model_type_to_workflow = {
    ModelType.SD15: SD15Workflow,
    ModelType.SDXL: SDXLWorkflow,
    ModelType.CASCADE: SDCascadeWorkflow,
    ModelType.PONY: PonyWorkflow,
    ModelType.SD3: SD3Workflow,
    ModelType.FLUX: FluxWorkflow
}

config = configparser.ConfigParser()
config.read("config.properties")
comfy_root_directory = config["LOCAL"]["COMFY_ROOT_DIR"]

loop = None


async def _do_txt2img(params: ImageWorkflow, model_type: ModelType, loras: list[Lora], interaction):
    with Workflow() as wf:
        workflow = model_type_to_workflow[model_type](params.model, params.clip_skip, loras, params.vae, params.use_tensorrt, params.tensorrt_model)
        workflow.create_latents(params.dimensions, params.batch_size)
        workflow.condition_prompts(params.prompt, params.negative_prompt)
        workflow.sample(params.seed, params.num_steps, params.cfg_scale, params.sampler, params.scheduler or "normal", use_ays=params.use_align_your_steps)
        images = workflow.decode_and_save("final_output")
    wf.task.add_preview_callback(lambda task, node_id, image: do_preview(task, node_id, image, interaction))
    results = await images
    await results
    image_batch = [await results.get(i) for i in range(params.batch_size)]
    return image_batch


async def _do_img2img(params: ImageWorkflow, model_type: ModelType, loras: list[Lora], interaction):
    with Workflow() as wf:
        workflow = model_type_to_workflow[model_type](params.model, params.clip_skip, loras, params.vae, params.use_tensorrt, params.tensorrt_model)
        image_input = LoadImage(params.filename)[0]
        workflow.create_img2img_latents(image_input, params.batch_size)
        if params.inpainting_prompt:
            workflow.mask_for_inpainting(image_input, params.inpainting_prompt, params.inpainting_detection_threshold)
        workflow.condition_prompts(params.prompt, params.negative_prompt)
        workflow.sample(params.seed, params.num_steps, params.cfg_scale, params.sampler, params.scheduler or "normal", params.denoise_strength, use_ays=params.use_align_your_steps)
        images = workflow.decode_and_save("final_output")
    wf.task.add_preview_callback(lambda task, node_id, image: do_preview(task, node_id, image, interaction))
    results = await images
    await results
    image_batch = [await results.get(i) for i in range(params.batch_size)]
    return image_batch


async def _do_upscale(params: ImageWorkflow, model_type: ModelType, loras: list[Lora], interaction):
    workflow = UpscaleWorkflow()
    workflow.load_image(params.filename)
    workflow.upscale(UPSCALE_DEFAULTS.model, 2.0)
    image = workflow.save("final_output")
    results = await image._wait()
    return await results.get(0)


async def _do_add_detail(params: ImageWorkflow, model_type: ModelType, loras: list[Lora], interaction):
    with Workflow() as wf:
        workflow = model_type_to_workflow[model_type](params.model, params.clip_skip, loras, params.vae, params.use_tensorrt, params.tensorrt_model)
        image_input = LoadImage(params.filename)[0]
        workflow.create_img2img_latents(image_input, params.batch_size)
        workflow.condition_prompts(params.prompt, params.negative_prompt)
        workflow.condition_for_detailing(params.detailing_controlnet, image_input)
        workflow.sample(params.seed, params.num_steps, params.cfg_scale, params.sampler, params.scheduler or "normal", params.denoise_strength, use_ays=False)
        images = workflow.decode_and_save("final_output")
    wf.task.add_preview_callback(lambda task, node_id, image: do_preview(task, node_id, image, interaction))
    results = await images
    await results
    image_batch = [await results.get(i) for i in range(params.batch_size)]
    return image_batch


async def _do_image_mashup(params: ImageWorkflow, model_type: ModelType, loras: list[Lora], interaction):
    with Workflow() as wf:
        workflow = model_type_to_workflow[model_type](params.model, params.clip_skip, loras, params.vae, params.use_tensorrt, params.tensorrt_model)
        image_inputs = [LoadImage(filename)[0] for filename in [params.filename, params.filename2] if filename is not None]
        workflow.create_latents(params.dimensions, params.batch_size)
        workflow.condition_prompts(params.prompt, params.negative_prompt)
        workflow.unclip_encode(image_inputs)
        workflow.sample(params.seed, params.num_steps, params.cfg_scale, params.sampler, params.scheduler or "normal", use_ays=params.use_align_your_steps)
        images = workflow.decode_and_save("final_output")
    wf.task.add_preview_callback(lambda task, node_id, image: do_preview(task, node_id, image, interaction))
    results = await images
    await results
    image_batch = [await results.get(i) for i in range(params.batch_size)]
    return image_batch


async def _do_video(params: ImageWorkflow, model_type: ModelType, loras: list[Lora], interaction):
    import PIL

    with open(params.filename, "rb") as f:
        image = PIL.Image.open(f)
        width = image.width
        height = image.height
        padding = 0
        if width / height <= 1:
            padding = height // 2

    with Workflow() as wf:
        image = LoadImage(params.filename)[0]
        image, _ = ImagePadForOutpaint(image, padding, 0, padding, 0, 40)
        model, clip_vision, vae = ImageOnlyCheckpointLoader(params.model)
        model = VideoLinearCFGGuidance(model, params.min_cfg)
        positive, negative, latent = SVDImg2vidConditioning(clip_vision, image, vae, 1024, 576, 25, params.motion, 8, params.augmentation)
        if config["VIDEO_GENERATION_DEFAULTS"]["USE_ALIGN_YOUR_STEPS"]:
            scheduler = AlignYourStepsScheduler("SVD", params.num_steps)
            sampler = KSamplerSelect("euler")
            latent, _ = SamplerCustom(model, True, params.seed, params.cfg_scale, positive, negative, sampler, scheduler, latent)
        else:
            latent = KSampler(model, params.seed, params.num_steps, params.cfg_scale, params.sampler, params.scheduler, positive, negative, latent, 1)
        image2 = VAEDecode(latent, vae)
        video = VHSVideoCombine(image2, 8, 0, "final_output", "image/gif", False, True, None, None)
        preview = PreviewImage(image)
    wf.task.add_preview_callback(lambda task, node_id, image: do_preview(task, node_id, image, interaction))
    await preview._wait()
    await video._wait()
    results = video.wait()._output
    final_video = PIL.Image.open(os.path.join(comfy_root_directory, "output", results["gifs"][0]["filename"]))
    return [final_video]


def process_prompt_with_llm(positive_prompt: str, seed: int, profile: str):
    from src.defaults import llm_prompt, llm_parameters

    prompt_text = llm_prompt + "\n" + positive_prompt
    _, prompt, _ = IFPromptMkr(
        input_prompt=prompt_text,
        engine=IFChatPrompt.engine.ollama,
        base_ip=llm_parameters["API_URL"],
        port=llm_parameters["API_PORT"],
        selected_model=llm_parameters["MODEL_NAME"],
        profile=profile,
        seed=seed,
        random=True,
    )
    return prompt


workflow_type_to_method = {
    WorkflowType.txt2img: _do_txt2img,
    WorkflowType.img2img: _do_img2img,
    WorkflowType.upscale: _do_upscale,
    WorkflowType.add_detail: _do_add_detail,
    WorkflowType.image_mashup: _do_image_mashup,
    WorkflowType.video: _do_video,
}

user_queues = {}


def do_preview(task, node_id, image, interaction):
    if image is None:
        return
    try:
        filename = f"temp_preview_{task.prompt_id}.png"
        fp = os.path.join(comfy_root_directory, "output", filename)
        image.save(fp)
        asyncio.run_coroutine_threadsafe(interaction.edit_original_response(attachments=[discord.File(fp, filename)]), loop)
    except Exception as e:
        print(e)


async def do_workflow(params: ImageWorkflow, interaction: discord.Interaction):
    global user_queues, loop
    loop = asyncio.get_event_loop()
    user = interaction.user

    if user_queues.get(user.id) is not None and user_queues[user.id] >= int(config["BOT"]["MAX_QUEUE_PER_USER"]):
        await interaction.edit_original_response(
            content=f"{user.mention} `You have too many pending requests. Please wait for them to finish. Amount in queue: {user_queues[user.id]}`"
        )
        return

    if user_queues.get(user.id) is None or user_queues[user.id] < 0:
        user_queues[user.id] = 0

    user_queues[user.id] += 1

    queue.watch_display(False, False, False)
    retries = 0
    while retries < MAX_RETRIES:
        try:
            loras = [Lora(lora, strength) for lora, strength in zip(params.loras, params.lora_strengths)] if params.loras else []

            extra_loras = get_loras_from_prompt(params.prompt)
            loras.extend([Lora(f"{lora[0]}.safetensors", lora[1]) for lora in extra_loras])

            if params.use_accelerator_lora and params.num_steps < 10:
                loras.append(Lora(params.accelerator_lora_name, 1.0))
                params.use_align_your_steps = False
            else:
                params.use_align_your_steps = True if params.model_type != ModelType.SD3 else False
                if params.cfg_scale < 1.2:
                    params.cfg_scale = 4.0

            if params.use_llm is True:
                enhanced_prompt = process_prompt_with_llm(params.prompt, params.seed, params.llm_profile)
                prompt_result = await IFDisplayText(enhanced_prompt)
                params.prompt = params.prompt + ", BREAK \n" + prompt_result._output["string"][0]

            if params.style_prompt is not None and params.style_prompt not in params.prompt:
                params.prompt = params.style_prompt + "\n" + params.prompt
            if params.negative_style_prompt is not None and (params.negative_prompt is None or params.negative_style_prompt not in params.negative_prompt):
                params.negative_prompt = params.negative_style_prompt + "\n" + (params.negative_prompt or "")

            params.style_prompt = None
            params.negative_style_prompt = None

            result = await workflow_type_to_method[params.workflow_type](params, params.model_type, loras, interaction)

            user_queues[user.id] -= 1
            await interaction.edit_original_response(attachments=[])
            return result
        except:
            user_queues[user.id] -= 1
            retries += 1

    raise Exception("Failed to generate image")
