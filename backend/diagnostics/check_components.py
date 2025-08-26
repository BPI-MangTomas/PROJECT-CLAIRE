import sys
import os
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

print("Testing components...")

# Test 1: Import modules
try:
    from app.core.answer_generator import AnswerGenerator
    print("✅ Imported AnswerGenerator")
except Exception as e:
    print(f"❌ Failed to import AnswerGenerator: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Create instance
try:
    gen = AnswerGenerator()
    print("✅ Created AnswerGenerator instance")
except Exception as e:
    print(f"❌ Failed to create AnswerGenerator: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test generate_answer
try:
    result = gen.generate_answer(
        question="test",
        language="english", 
        emotion="neutral",
        contexts=[{"content": "test", "score": 0.9}],
        extracted_text=None
    )
    print(f"✅ Generated answer: {result}")
except Exception as e:
    print(f"❌ Failed to generate answer: {e}")
    import traceback
    traceback.print_exc()