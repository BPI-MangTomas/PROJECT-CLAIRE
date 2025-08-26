# check_structure.py
import os

print("Checking backend/app/core directory structure...")
print("=" * 50)

core_path = os.path.join("backend", "app", "core")

if os.path.exists(core_path):
    files = os.listdir(core_path)
    print(f"Files in {core_path}:")
    for file in sorted(files):
        if file.endswith('.py'):
            print(f"  ✓ {file}")
else:
    print(f"❌ Directory not found: {core_path}")

# Try alternative path
alt_path = os.path.join("app", "core")
if os.path.exists(alt_path):
    print(f"\nFiles in {alt_path}:")
    files = os.listdir(alt_path)
    for file in sorted(files):
        if file.endswith('.py'):
            print(f"  ✓ {file}")