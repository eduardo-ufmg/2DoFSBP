# -*- coding: utf-8 -*-
"""
Generates a KiCad footprint for a standard 2.54mm perfboard.
"""

# --- Configuration ---
ROWS = 27
COLS = 35
PITCH = 2.54  # in mm
DRILL_DIA = 1.0  # in mm
PAD_DIA = 1.8  # in mm
FOOTPRINT_NAME = f"Perfboard_{ROWS}x{COLS}_{PITCH:.2f}mm"
OUTPUT_FILENAME = f"{FOOTPRINT_NAME}.kicad_mod"


def generate_footprint():
    """
    Main function to generate the KiCad module file content.
    """
    print(f"Generating footprint '{FOOTPRINT_NAME}'...")

    # Calculate the total width and height to center the footprint
    total_width = (COLS - 1) * PITCH
    total_height = (ROWS - 1) * PITCH
    x_offset = total_width / 2.0
    y_offset = total_height / 2.0

    # --- Start of the KiCad Module File ---
    # The (fp_text ...), (fp_line ...), etc. are optional but good for documentation.
    file_content = [
        f"(module {FOOTPRINT_NAME} (layer F.Cu) (tedit 5B307E46)",
        "  (descr \"Perfboard, {ROWS}x{COLS} grid, {PITCH:.2f}mm pitch, "
        f"{PAD_DIA:.2f}mm pads, {DRILL_DIA:.2f}mm drills\")",
        "  (tags \"perfboard protoboard\")",
        f"  (fp_text reference REF** (at 0 {-(y_offset + 2)}) (layer F.SilkS) (effects (font (size 1 1) (thickness 0.15))))",
        f"  (fp_text value {FOOTPRINT_NAME} (at 0 {y_offset + 2}) (layer F.Fab) (effects (font (size 1 1) (thickness 0.15))))",
        # Add a bounding box on the fabrication layer for clarity
        f"  (fp_line (start {-x_offset - PITCH/2} {-y_offset - PITCH/2}) (end {x_offset + PITCH/2} {-y_offset - PITCH/2}) (layer F.Fab) (width 0.15))",
        f"  (fp_line (start {x_offset + PITCH/2} {-y_offset - PITCH/2}) (end {x_offset + PITCH/2} {y_offset + PITCH/2}) (layer F.Fab) (width 0.15))",
        f"  (fp_line (start {x_offset + PITCH/2} {y_offset + PITCH/2}) (end {-x_offset - PITCH/2} {y_offset + PITCH/2}) (layer F.Fab) (width 0.15))",
        f"  (fp_line (start {-x_offset - PITCH/2} {y_offset + PITCH/2}) (end {-x_offset - PITCH/2} {-y_offset - PITCH/2}) (layer F.Fab) (width 0.15))"
    ]

    # --- Generate Pads ---
    pin_number = 1
    for row in range(ROWS):
        for col in range(COLS):
            # Calculate pad position relative to the center
            x_pos = col * PITCH - x_offset
            y_pos = row * PITCH - y_offset

            # Format the pad string using KiCad's S-expression syntax
            # (pad <number> thru_hole circle (at <x> <y>) (size <w> <h>) (drill <d>) (layers *.Cu *.Mask))
            pad_string = (
                f"  (pad {pin_number} thru_hole circle (at {x_pos:.4f} {y_pos:.4f}) "
                f"(size {PAD_DIA:.4f} {PAD_DIA:.4f}) (drill {DRILL_DIA:.4f}) "
                "(layers *.Cu *.Mask))"
            )
            file_content.append(pad_string)
            pin_number += 1

    # --- End of File ---
    file_content.append(")")

    # --- Write to File ---
    try:
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            f.write("\n".join(file_content))
        print(f"✅ Successfully created file: {OUTPUT_FILENAME}")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")


if __name__ == "__main__":
    generate_footprint()