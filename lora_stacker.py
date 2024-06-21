from dataclasses import dataclass
import folder_paths


@dataclass
class LoraItem:
    name: str
    weight: float


StackedLora = list[LoraItem]


class LoraStacker:
    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "lora_count": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
            }
        }

        for i in range(1, 50):
            inputs["required"][f"lora_name_{i}"] = (loras,)
            inputs["required"][f"lora_weight_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
            )

        return inputs

    RETURN_TYPES = ("TRT_LORA_STACK",)
    RETURN_NAMES = ("TRT_LORA_STACK",)
    FUNCTION = "lora_stacker"
    CATEGORY = "TorchTensorRT/LoRA stacker"

    def lora_stacker(
        self, lora_count: int, **kwargs: str | float
    ) -> tuple[StackedLora,]:
        # Extract values from kwargs
        lora_names: list[str] = [
            str(kwargs.get(f"lora_name_{i}")) for i in range(1, lora_count + 1)
        ]

        # Create a list of tuples using provided parameters, exclude tuples with lora_name as "None"
        weights: list[float] = [
            float(str(kwargs.get(f"lora_weight_{i}"))) for i in range(1, lora_count + 1)
        ]
        loras = [
            LoraItem(lora_name, lora_weight)
            for lora_name, lora_weight in zip(lora_names, weights)
            if lora_name != "None"
        ]

        return (loras,)
