#!/bin/bash
# Train all 5 VRCI models
# Contact: admin@gy4k.com

echo "🚀 VRCI Model Training Pipeline"
echo "==============================="
echo ""

# Check if training data exists
if [ ! -d "../../training_data" ]; then
    echo "⚠️  Training data not found. Generating..."
    python generate_training_data.py
fi

echo "📊 Training data ready!"
echo ""

# Train each model
models=("latency" "energy" "coverage" "consensus" "carbon")

for model in "${models[@]}"; do
    echo "🔧 Training $model model..."
    python train_${model}_model.py
    
    if [ $? -eq 0 ]; then
        echo "✅ $model model trained successfully!"
    else
        echo "❌ $model model training failed!"
        exit 1
    fi
    echo ""
done

echo "🎉 All models trained successfully!"
echo "📁 Model checkpoints saved to: ../models/"
echo ""
echo "Next step: Run platform with ./start_platform.sh"
