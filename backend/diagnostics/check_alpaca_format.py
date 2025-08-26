"""
Test script to verify the Alpaca format works correctly with CLAIRE GGUF model
"""

from llama_cpp import Llama
import json

def test_alpaca_format():
    """Test the exact Alpaca format used in training"""
    
    print("Loading CLAIRE model...")
    model = Llama(
        model_path="./backend/models/claire_v1.0.0_q4_k_m.gguf",
        n_ctx=2048,
        n_gpu_layers=0,  # Set to 35 or higher if using GPU
        verbose=False
    )
    
    # Test case matching training format
    question = "How can I reset my BPI online banking password?"
    language = "english"
    emotion = "worried"
    
    # Sample contexts (as they would appear from your retrieval system)
    contexts = [
        {
            "content": "To reset your BPI online banking password, visit the BPI website and click on 'Forgot Password'. You'll need your username and registered email or mobile number. Follow the instructions sent to your registered contact to create a new password.",
            "score": 0.92
        },
        {
            "content": "For security reasons, passwords must be at least 8 characters long and include uppercase, lowercase, numbers, and special characters. Change your password regularly for better security.",
            "score": 0.75
        },
        {
            "content": "If you're unable to reset your password online, visit any BPI branch with a valid ID or call our 24/7 hotline at 889-10000 for assistance.",
            "score": 0.68
        },
        {
            "content": "BPI online banking provides secure access to your accounts. Always ensure you're on the official BPI website before entering your credentials.",
            "score": 0.45
        }
    ]
    
    # Format contexts exactly as in training
    context_texts = []
    for i, ctx in enumerate(contexts):
        context_texts.append(f"Context {i+1} (Score: {ctx['score']:.2f}): {ctx['content']}")
    
    formatted_contexts = "\n\n".join(context_texts)
    
    # Create the EXACT prompt format used in training
    prompt = (
        f"### Instruction:\n"
        f"You are CLAIRE (Conversational Language AI for Resolution & Engagement), "
        f"a banking customer assistant working for BPI (Bank of the Philippine Islands). "
        f"Your role is to answer customer questions accurately, clearly, and empathetically. "
        f"Given the question, its identified language and emotion, and four context documents, "
        f"generate a response that is linguistically accurate, emotionally appropriate, "
        f"and grounded in the most relevant context.\n\n"
        f"### Input:\n"
        f"Question: {question}\n"
        f"Language: {language}\n"
        f"Emotion: {emotion}\n\n"
        f"Contexts:\n{formatted_contexts}\n\n"
        f"### Output:\n"
    )
    
    print("\n" + "="*60)
    print("TESTING ALPACA FORMAT")
    print("="*60)
    print(f"Question: {question}")
    print(f"Language: {language}")
    print(f"Emotion: {emotion}")
    print("-"*60)
    
    # Generate response
    response = model(
        prompt,
        max_tokens=200,
        temperature=0.4,
        top_p=0.9,
        echo=False,
        stop=["### Instruction:", "### Input:", "### Output:", "\n\n### "],
        repeat_penalty=1.1
    )
    
    generated = response['choices'][0]['text'].strip()
    
    print("\nGenerated Response:")
    print("-"*60)
    print(generated)
    print("-"*60)
    
    # Analyze response quality
    print("\nResponse Analysis:")
    print("-"*60)
    
    # Check for relevant content
    relevant_keywords = ["password", "reset", "online banking", "889-10000", "branch", "website"]
    found_keywords = [kw for kw in relevant_keywords if kw.lower() in generated.lower()]
    print(f"✓ Relevant keywords found: {', '.join(found_keywords) if found_keywords else 'None'}")
    
    # Check for emotion handling (worried)
    emotion_phrases = ["understand", "help", "assist", "worry", "concern", "support"]
    has_emotion = any(phrase in generated.lower() for phrase in emotion_phrases)
    print(f"✓ Emotion handling: {'Yes' if has_emotion else 'No'}")
    
    # Check for artifacts
    artifacts = ["###", "Context 1", "Context 2", "(Score:", "### Instruction", "### Input", "### Output"]
    found_artifacts = [art for art in artifacts if art in generated]
    if found_artifacts:
        print(f"⚠ Artifacts found: {', '.join(found_artifacts)}")
    else:
        print("✓ No artifacts in response")
    
    # Check response length
    print(f"✓ Response length: {len(generated)} characters")
    
    return generated


def test_different_scenarios():
    """Test multiple scenarios to verify consistency"""
    
    print("\n" + "="*60)
    print("TESTING MULTIPLE SCENARIOS")
    print("="*60)
    
    model = Llama(
        model_path="./backend/models/claire_v1.0.0_q4_k_m.gguf",
        n_ctx=2048,
        n_gpu_layers=0,
        verbose=False
    )
    
    test_cases = [
        {
            "question": "What are BPI's banking hours?",
            "language": "english",
            "emotion": "neutral",
            "context": "BPI branches are typically open from 9:00 AM to 3:00 PM on weekdays."
        },
        {
            "question": "Ano ang banking hours ng BPI?",
            "language": "tagalog", 
            "emotion": "neutral",
            "context": "Ang mga sangay ng BPI ay karaniwang bukas mula 9:00 AM hanggang 3:00 PM tuwing weekdays."
        },
        {
            "question": "Paano mag-open ng savings account? Thank youu",
            "language": "taglish",
            "emotion": "grateful",
            "context": "To open a BPI savings account, bring 2 valid IDs and PHP 500 initial deposit to any branch."
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['language'].title()} - {test['emotion']}")
        print(f"Question: {test['question']}")
        
        # Create contexts (simulate retrieval with one main context and fillers)
        contexts_formatted = f"Context 1 (Score: 0.95): {test['context']}\n\n"
        contexts_formatted += "Context 2 (Score: 0.30): General BPI information.\n\n"
        contexts_formatted += "Context 3 (Score: 0.25): BPI contact details.\n\n"
        contexts_formatted += "Context 4 (Score: 0.20): BPI services overview."
        
        prompt = (
            f"### Instruction:\n"
            f"You are CLAIRE (Conversational Language AI for Resolution & Engagement), "
            f"a banking customer assistant working for BPI (Bank of the Philippine Islands). "
            f"Your role is to answer customer questions accurately, clearly, and empathetically. "
            f"Given the question, its identified language and emotion, and four context documents, "
            f"generate a response that is linguistically accurate, emotionally appropriate, "
            f"and grounded in the most relevant context.\n\n"
            f"### Input:\n"
            f"Question: {test['question']}\n"
            f"Language: {test['language']}\n"
            f"Emotion: {test['emotion']}\n\n"
            f"Contexts:\n{contexts_formatted}\n\n"
            f"### Output:\n"
        )
        
        response = model(
            prompt,
            max_tokens=150,
            temperature=0.4,
            echo=False,
            stop=["### Instruction:", "### Input:", "### Output:", "\n\n### "]
        )
        
        generated = response['choices'][0]['text'].strip()
        print(f"Response: {generated[:100]}...[truncated]")
        print("-"*40)


if __name__ == "__main__":
    # Run main test
    test_alpaca_format()
    
    # Run additional scenarios
    test_different_scenarios()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("""
If the responses:
1. ✓ Answer the questions correctly
2. ✓ Match the requested language
3. ✓ Reflect the emotion appropriately  
4. ✓ Don't contain format artifacts (###, Context 1, etc.)
5. ✓ Are coherent and complete

Then the Alpaca format is working correctly!
""")