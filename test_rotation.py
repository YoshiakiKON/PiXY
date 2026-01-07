"""Quick test to verify view_orientation logic"""

# Simulate the condition check
view_orientation = 'Stage'
info = {'angle_deg': 45.0, 's': 1.0}  # Simulated info

print(f"view_orientation = '{view_orientation}'")
print(f"info is not None = {info is not None}")
print(f"view_orientation == 'Stage' = {view_orientation == 'Stage'}")
print(f"Condition result: {info is not None and view_orientation == 'Stage'}")

# Test getattr
class TestObj:
    pass

obj = TestObj()
obj.view_orientation = 'Stage'

view_orient = getattr(obj, 'view_orientation', 'Image')
print(f"\ngetattr result: '{view_orient}'")
print(f"getattr == 'Stage': {view_orient == 'Stage'}")
