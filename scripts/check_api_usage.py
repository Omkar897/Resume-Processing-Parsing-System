"""
Check Anthropic API usage and credits
"""

import os
from dotenv import load_dotenv

load_dotenv()


def check_anthropic_credits():
    """Check remaining credits and usage"""
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in .env")
        return

    print("=" * 60)
    print("ANTHROPIC API USAGE CHECK")
    print("=" * 60)

    print(f"\n✅ API Key found: {api_key[:15]}...{api_key[-4:]}")
    print("\n📊 To check your usage and credits:")
    print("   1. Visit: https://console.anthropic.com")
    print("   2. Go to 'Usage' section")
    print("   3. View your remaining credits and usage stats")

    print("\n💡 Quick Stats:")
    print("   - Model: Claude 3 Haiku")
    print("   - Cost per resume analysis: ~$0.0004")
    print("   - With $5 credit: ~12,500 resumes")

    print("\n🔗 Useful Links:")
    print("   - Console: https://console.anthropic.com")
    print("   - Pricing: https://www.anthropic.com/pricing")
    print("   - Docs: https://docs.anthropic.com")
    print("=" * 60)


if __name__ == "__main__":
    check_anthropic_credits()
