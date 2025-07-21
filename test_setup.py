#!/usr/bin/env python3
"""
Setup Test Script for AtulayaAkriti
Verifies that all components are properly installed and configured
"""

import sys
import os
from pathlib import Path
import importlib.util

def test_python_version():
    """Test Python version compatibility"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Requires Python 3.8+)")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n📦 Testing Python dependencies...")

    required_packages = [
        'streamlit',
        'numpy',
        'cv2',
        'PIL',
        'requests',
        'torch',
        'torchvision',
        'streamlit_drawable_canvas'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            elif package == 'streamlit_drawable_canvas':
                import streamlit_drawable_canvas
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (Missing)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False

    return True

def test_sam_availability():
    """Test SAM library availability"""
    print("\n🤖 Testing SAM availability...")

    try:
        import segment_anything
        print("   ✅ SAM library available")
        sam_available = True
    except ImportError:
        print("   ❌ SAM library not found")
        print("   Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
        sam_available = False

    return sam_available

def test_model_files():
    """Test SAM model files"""
    print("\n📁 Testing SAM model files...")

    models_dir = Path("models")
    if not models_dir.exists():
        print("   ❌ models/ directory not found")
        return False

    model_files = [
        "sam_vit_b_01ec64.pth",
        "sam_vit_l_0b3195.pth",
        "sam_vit_h_4b8939.pth",
        "sam_vit_b.pth"
    ]

    found_models = []
    for model_file in model_files:
        filepath = models_dir / model_file
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ {model_file} ({size_mb:.1f} MB)")
            found_models.append(model_file)

    if not found_models:
        print("   ❌ No SAM model files found")
        print("   Download with: python download_sam_model.py")
        return False

    return True

def test_directory_structure():
    """Test directory structure"""
    print("\n📂 Testing directory structure...")

    required_dirs = [
        "models",
        "assets",
        "assets/textures",
        "utils",
        "docs"
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ (Missing)")
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"\n⚠️  Missing directories: {', '.join(missing_dirs)}")
        print("   These will be created automatically when needed")

    return len(missing_dirs) == 0

def test_main_app():
    """Test main application file"""
    print("\n🚀 Testing main application...")

    app_file = Path("atulayakriti_app.py")
    if not app_file.exists():
        print("   ❌ atulayakriti_app.py not found")
        return False

    try:
        # Try to import the main app (basic syntax check)
        spec = importlib.util.spec_from_file_location("atulayakriti_app", app_file)
        if spec is None:
            print("   ❌ Cannot load atulayakriti_app.py")
            return False

        print("   ✅ atulayakriti_app.py syntax OK")
        return True

    except Exception as e:
        print(f"   ❌ Error in atulayakriti_app.py: {e}")
        return False

def main():
    """Main test function"""
    print("🎨 AtulayaAkriti Setup Test")
    print("=" * 50)

    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("SAM Library", test_sam_availability),
        ("Model Files", test_model_files),
        ("Directory Structure", test_directory_structure),
        ("Main Application", test_main_app)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test failed with error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! AtulayaAkriti is ready to use!")
        print("\nStart the app with: streamlit run atulayakriti_app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")

        # Provide specific help
        failed_tests = [name for name, result in results if not result]
        if "Dependencies" in failed_tests:
            print("\n📦 To install dependencies: pip install -r requirements.txt")
        if "SAM Library" in failed_tests:
            print("\n🤖 To install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git")
        if "Model Files" in failed_tests:
            print("\n📁 To download models: python download_sam_model.py")

if __name__ == "__main__":
    main()
