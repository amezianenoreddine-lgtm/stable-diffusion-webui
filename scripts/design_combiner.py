import argparse
import base64
import json
import os
import statistics
from collections import Counter

import requests
from PIL import Image


def to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path)


def get_image_metadata(image: Image.Image) -> dict:
    image = image.convert("RGB")
    width, height = image.size
    pixels = list(image.getdata())
    brightness = statistics.mean((sum(pixel) / 3) for pixel in pixels)
    contrast = statistics.pstdev((sum(pixel) / 3) for pixel in pixels)
    grayscale = all(r == g == b for (r, g, b) in pixels[: min(5000, len(pixels))])
    dominant_colors = get_dominant_colors(image, count=4)
    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 3) if height else None,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "grayscale": grayscale,
        "dominant_colors": dominant_colors,
    }


def get_dominant_colors(image: Image.Image, count: int = 4) -> list[str]:
    small = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    palette = small.quantize(colors=count, method=Image.MEDIANCUT)
    colors = palette.getcolors(128 * 128)
    if not colors:
        return []
    colors = sorted(colors, reverse=True)
    palette_colors = palette.getpalette()
    hex_colors = []
    for _, index in colors[:count]:
        offset = index * 3
        rgb = palette_colors[offset:offset + 3]
        if len(rgb) == 3:
            hex_colors.append("#{:02x}{:02x}{:02x}".format(*rgb))
    return hex_colors


def analyze_image_style(image_path: str, api_url: str, model: str = "clip") -> dict:
    image = load_image(image_path)
    metadata = get_image_metadata(image)
    caption = interrogate_image(image_path, api_url, model=model)
    style_tags = []
    if metadata["grayscale"]:
        style_tags.append("monochrome graphic")
    elif metadata["contrast"] > 55:
        style_tags.append("high contrast artwork")

    if metadata["aspect_ratio"] and metadata["aspect_ratio"] > 1.3:
        style_tags.append("horizontal layout")
    elif metadata["aspect_ratio"] and metadata["aspect_ratio"] < 0.8:
        style_tags.append("vertical layout")
    else:
        style_tags.append("centered badge layout")

    if any(color.startswith("#f") for color in metadata["dominant_colors"][:2]):
        style_tags.append("warm vintage palette")
    elif any(color.startswith("#0") for color in metadata["dominant_colors"][:2]):
        style_tags.append("dark and moody palette")

    return {
        "path": image_path,
        "caption": caption,
        "metadata": metadata,
        "style_tags": style_tags,
    }


def interrogate_image(image_path: str, api_url: str, model: str = "clip") -> str:
    payload = {"image": to_base64(image_path), "model": model}
    response = requests.post(f"{api_url}/sdapi/v1/interrogate", json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("caption", "")


def build_combined_prompt(
    concept_caption: str,
    style_caption: str,
    concept_text: str | None,
    style_text: str | None,
) -> tuple[str, str]:
    concept_phrase = concept_text or concept_caption or "a bold liquor-themed character design"
    style_phrase = style_text or style_caption or "a vintage badge engraving style"

    prompt = (
        f"High quality t-shirt graphic design, print-ready, transparent background, centered composition, "
        f"concept: {concept_phrase}, style: {style_phrase}, "
        f"keep the concept dominant and literal while using the style as a refined retro treatment, "
        f"engraved line art texture, strong contrast, bold serif lettering, ornamental badge details, "
        f"market-competitive vintage t-shirt design, vector-friendly look"
    )

    negative_prompt = (
        "low quality, blurry, extra limbs, watermark, photographic realism, "
        "digital painting, cartoonish, off-center, cluttered background, distorted text, "
        "over-saturated colors, ugly, poorly drawn, low resolution"
    )

    return prompt, negative_prompt


def generate_graphic(
    api_url: str,
    prompt: str,
    negative_prompt: str,
    output_path: str,
    concept_image_path: str | None = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 40,
    cfg_scale: float = 11.0,
    denoising_strength: float = 0.55,
    sampler: str = "Euler a",
) -> dict:
    request_body = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler_index": sampler,
        "send_images": True,
        "save_images": False,
    }

    if concept_image_path:
        request_body["init_images"] = [to_base64(concept_image_path)]
        request_body["denoising_strength"] = denoising_strength
        url = f"{api_url}/sdapi/v1/img2img"
    else:
        url = f"{api_url}/sdapi/v1/txt2img"

    response = requests.post(url, json=request_body, timeout=180)
    response.raise_for_status()
    result = response.json()

    images = result.get("images", [])
    if not images:
        raise RuntimeError("No image returned from Stable Diffusion API")

    image_data = base64.b64decode(images[0].split(",")[-1])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(image_data)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze concept/style designs and generate a combined t-shirt graphic design.",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:7860", help="Stable Diffusion WebUI API URL")
    parser.add_argument("--concept-image", help="Path to the concept design image")
    parser.add_argument("--style-image", help="Path to the style reference image")
    parser.add_argument("--concept-text", help="Optional concept text prompt")
    parser.add_argument("--style-text", help="Optional style text prompt")
    parser.add_argument("--output", default="combined_design.png", help="Output file path for the generated design")
    parser.add_argument("--width", type=int, default=1024, help="Output image width")
    parser.add_argument("--height", type=int, default=1024, help="Output image height")
    parser.add_argument("--steps", type=int, default=40, help="Number of sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=11.0, help="CFG scale")
    parser.add_argument("--denoising-strength", type=float, default=0.55, help="Denoising strength for img2img")
    parser.add_argument("--sampler", default="Euler a", help="Sampler name for generation")
    parser.add_argument("--analysis-only", action="store_true", help="Only analyze inputs and print the prompt, do not generate an image")

    args = parser.parse_args()

    if not args.concept_image and not args.concept_text:
        parser.error("At least one of --concept-image or --concept-text is required.")

    if args.concept_image and not os.path.exists(args.concept_image):
        parser.error(f"Concept image not found: {args.concept_image}")

    if args.style_image and not os.path.exists(args.style_image):
        parser.error(f"Style image not found: {args.style_image}")

    concept_analysis = None
    style_analysis = None

    if args.concept_image:
        concept_analysis = analyze_image_style(args.concept_image, args.api_url)
        print("Concept analysis:")
        print(json.dumps(concept_analysis, indent=2))

    if args.style_image:
        style_analysis = analyze_image_style(args.style_image, args.api_url)
        print("Style analysis:")
        print(json.dumps(style_analysis, indent=2))

    prompt, negative_prompt = build_combined_prompt(
        concept_caption=(args.concept_text or (concept_analysis and concept_analysis["caption"])),
        style_caption=(args.style_text or (style_analysis and style_analysis["caption"])),
        concept_text=args.concept_text,
        style_text=args.style_text,
    )

    print("Generated prompt:")
    print(prompt)
    print("Negative prompt:")
    print(negative_prompt)

    if args.analysis_only:
        return

    result = generate_graphic(
        api_url=args.api_url,
        prompt=prompt,
        negative_prompt=negative_prompt,
        output_path=args.output,
        concept_image_path=args.concept_image,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        denoising_strength=args.denoising_strength,
        sampler=args.sampler,
    )

    print(f"Generated image saved to {args.output}")
    print("Generation info:")
    print(json.dumps(result.get("info", {}), indent=2))


if __name__ == "__main__":
    main()
