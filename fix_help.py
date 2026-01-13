
import os

file_path = r"c:\Users\HP\OneDrive\Desktop\image ython\app.py"
new_help_text = '                help="To control how much is selected, choose a mode:\\n\\n🏠 **Big Surfaces:** Selects the **WHOLE** connected wall/floor. Best for painting entire rooms.\\n\\n⚡ **Auto-Select:** Smart balance. Best for general use.\\n\\n🎯 **Small Details:** Selects only the specific sub-part you clicked (e.g. just a cabinet door, not the whole kitchen)."\n'

try:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    new_lines = []
    found = False
    for line in lines:
        if 'help="' in line and '**Big Surfaces:**' in line:
            new_lines.append(new_help_text)
            found = True
        else:
            new_lines.append(line)
            
    if found:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print("Successfully updated app.py")
    else:
        print("Target line not found.")

except Exception as e:
    print(f"Error: {e}")
