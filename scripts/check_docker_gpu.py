#!/usr/bin/env python3
"""
Docker内でのGPU/MPS利用可能性チェックスクリプト
"""


def check_pytorch_devices():
    print("🔍 PyTorch GPU/MPS support check")
    print("=" * 50)

    try:
        import torch

        print(f"✅ PyTorch version: {torch.__version__}")

        # CUDA support
        print(f"🔥 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

        # MPS (Apple Silicon) support
        if hasattr(torch.backends, "mps"):
            print(f"🍎 MPS available: {torch.backends.mps.is_available()}")
            print(f"🍎 MPS built: {torch.backends.mps.is_built()}")
        else:
            print("🍎 MPS: Not supported in this PyTorch version")

        # CPU info
        print(f"💻 CPU device: {torch.device('cpu')}")

        # Test tensor creation on available devices
        print("\n🧪 Device test:")

        # CPU test
        cpu_tensor = torch.randn(3, 3)
        print(f"   ✅ CPU tensor: {cpu_tensor.device}")

        # MPS test
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                mps_tensor = torch.randn(3, 3, device="mps")
                print(f"   ✅ MPS tensor: {mps_tensor.device}")
            except Exception as e:
                print(f"   ❌ MPS tensor failed: {e}")

        # CUDA test
        if torch.cuda.is_available():
            try:
                cuda_tensor = torch.randn(3, 3, device="cuda")
                print(f"   ✅ CUDA tensor: {cuda_tensor.device}")
            except Exception as e:
                print(f"   ❌ CUDA tensor failed: {e}")

    except ImportError:
        print("❌ PyTorch not available")


def check_autogluon_gpu():
    print("\n🤖 AutoGluon GPU support check")
    print("=" * 50)

    try:
        from autogluon.timeseries import TimeSeriesPredictor

        print("✅ AutoGluon TimeSeries imported successfully")

        # GPUサポート情報を取得
        try:
            predictor = TimeSeriesPredictor(prediction_length=1, target="target")
            print("✅ TimeSeriesPredictor created successfully")

            # 利用可能なモデル一覧
            available_models = predictor._get_model_defaults()
            print(f"📋 Available models: {list(available_models.keys())}")

            # Chronosモデルの確認
            if "Chronos" in available_models:
                print("✅ Chronos model available")
                chronos_config = available_models["Chronos"]
                print(f"   Default config: {chronos_config}")
            else:
                print("❌ Chronos model not found")

        except Exception as e:
            print(f"❌ AutoGluon initialization failed: {e}")

    except ImportError as e:
        print(f"❌ AutoGluon not available: {e}")


def check_huggingface_transformers():
    print("\n🤗 HuggingFace Transformers GPU check")
    print("=" * 50)

    try:
        import transformers

        print(f"✅ Transformers version: {transformers.__version__}")

        try:
            import torch

            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"🎯 Recommended device: {device}")

            # Chronosモデル情報
            chronos_models = [
                "amazon/chronos-t5-tiny",
                "amazon/chronos-t5-mini",
                "amazon/chronos-t5-small",
                "amazon/chronos-t5-base",
            ]

            print("📋 Available Chronos models:")
            for model in chronos_models:
                print(f"   - {model}")

        except Exception as e:
            print(f"❌ Device detection failed: {e}")

    except ImportError:
        print("❌ Transformers not available")


def main():
    print("🐳 Docker GPU compatibility check")
    print("=" * 60)

    check_pytorch_devices()
    check_autogluon_gpu()
    check_huggingface_transformers()

    print("\n" + "=" * 60)
    print("🏁 Check completed")


if __name__ == "__main__":
    main()
