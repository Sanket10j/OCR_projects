from typing import Optional
import asyncio
import logging
import os

from python3_capsolver.core.captcha_instrument import FileInstrument
from python3_capsolver.image_to_text import ImageToText

async def solve_captcha_image(
    logger: logging.Logger,
    image_path: str,
    max_retries: int,
    type_of_captcha: str = 'common'
) -> Optional[str]:
    """
    Solve the CAPTCHA using the external CAPSOLVER API.

    Args:
        logger: Logger instance for logging
        image_path: Path to the CAPTCHA image file
        max_retries: Number of attempts to try solving the CAPTCHA
        type_of_captcha: Type/module of CAPTCHA (e.g., 'common', 'simple-image')

    Returns:
        Solved CAPTCHA text, or None if unsuccessful
    """
    logger.debug(f"Solving CAPTCHA from image: {image_path}")

    # Set API key (you can also load this from environment variable securely)
    api_key = "CAP-DD7595E6BEFBC7EC442ECD2E52CE90B707C069A9774022B9706880D71F56E60D"
    if not api_key:
        raise ValueError("CAPSOLVER_API_KEY is not set")

    # Validate image file
    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        raise FileNotFoundError(f"CAPTCHA image file not found: {image_path}")

    if os.path.getsize(image_path) < 100:
        raise ValueError(f"CAPTCHA image file too small: {image_path}")

    # Convert image to base64 body
    body = FileInstrument().file_processing(captcha_file=image_path)
    logger.debug("BODY of captcha: %s", body)

    # Prepare solver
    solver = ImageToText(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = await solver.aio_captcha_handler(
                task_payload={"body": body, "module": type_of_captcha},
            )
            logger.debug("CAPTCHA solver raw response: %s", response)

            # Check if a valid solution was returned
            if response.get('status') == 'ready':
                captcha_text = response.get('solution', {}).get('text')
                logger.warning("TEXT of captcha: %s", captcha_text)
                if captcha_text:
                    logger.warning("CAPTCHA solved successfully")
                    logger.debug("CAPTCHA solved: %s", captcha_text)
                    return captcha_text
                else:
                    logger.warning("Empty CAPTCHA solution returned")
            else:
                error_detail = response.get('errorDescription', 'Unknown error')
                logger.warning("Solver error: %s (Attempt %d)", error_detail, attempt + 1)

        except Exception as e:
            logger.warning("Solver communication error (Attempt %d): %s", attempt + 1, str(e))

        if attempt < max_retries - 1:
            await asyncio.sleep(1 + attempt)

    return None
