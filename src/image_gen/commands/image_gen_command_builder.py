# represents a custom image gen command
class image_gen_command:
    def __init__(self):
        self.name = None
        self.prompt = None
        self.lora1 = None
        self.lora1strength = None
        self.clipskip = None

    def __str__(self):
        return f"image_gen_command {self.name} with \"{self.prompt}\" as a prompt, {self.lora1} as a lora, and {self.lora1strength} as that lora's strength, and clip skip {self.clipskip}"
    
# base class for a command, not meant to be implemented
class image_gen_command_builder:
    def build_name(self):
        raise NotImplementedError

    def build_prompt(self):
        raise NotImplementedError

    def build_lora1(self):
        raise NotImplementedError
    
    def build_clipskip(self):
        raise NotImplementedError

    def get_imagegencommand(self):
        raise NotImplementedError
    
# feed zach example implementation in code
class feed_zach_image_gen_command_builder(image_gen_command_builder):
    def __init__(self):
        self.command = image_gen_command()

    def build_name(self):
        self.command.name = "FeedZach"

    def build_prompt(self):
        self.command.prompt = "a picture of feedzach smiling"

    def build_lora(self):
        self.command.lora1 = "feedzach-xl"

    def build_lora_strength(self):
        self.command.lora_strength = 1

    def build_clip_skip(self):
        self.command.clip_skip = -2
    
    def build_sdxl(self):
        self.command.sdxl = True 

    def get_command(self):
        return self.command