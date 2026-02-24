def generate_safety_flags(
    daily_dose_mg=None,
    max_daily_dose_mg=None,
    dose_per_administration_mg=None,
    withdrawal_volume_ml=None
):
    """
    Generates safety flags based on available dosing parameters.
    All parameters are optional to allow flexible clinical workflows.
    """

    flags = []

    # --- Daily dose safety ---
    if daily_dose_mg is not None and max_daily_dose_mg is not None:
        if daily_dose_mg > max_daily_dose_mg:
            flags.append(
                f"Total daily dose ({daily_dose_mg:.2f} mg) exceeds "
                f"maximum recommended daily dose ({max_daily_dose_mg:.2f} mg)."
            )

    # --- Single dose safety ---
    if dose_per_administration_mg is not None:
        if dose_per_administration_mg > 1000:
            flags.append(
                f"Single dose ({dose_per_administration_mg:.2f} mg) is high. "
                "Verify indication and administration rate."
            )

    # --- Withdrawal volume safety ---
    if withdrawal_volume_ml is not None:
        if withdrawal_volume_ml > 10:
            flags.append(
                f"Withdrawal volume ({withdrawal_volume_ml:.2f} mL) is large. "
                "Consider further dilution or alternative vial strength."
            )

    if not flags:
        flags.append("No immediate safety concerns detected.")

    return flags


def format_safety_comment(flags):
    """
    Formats safety flags into user-readable output.
    """
    formatted = []
    for flag in flags:
        formatted.append(f"• {flag}")
    return formatted
