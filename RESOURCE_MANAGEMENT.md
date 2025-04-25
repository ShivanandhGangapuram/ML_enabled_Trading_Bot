# Resource Management in Trading Bot

This document explains how the Trading Bot manages system resources to ensure efficient operation without overwhelming your computer.

## Machine Learning Training

The machine learning component of the Trading Bot is designed to be resource-aware, allowing you to control how much of your system's CPU and memory it uses during training.

### CPU Usage Control

You can select from four CPU usage levels:

1. **Low (25%)**: Uses approximately 25% of available CPU cores. Best for when you need to use your computer for other tasks while training.

2. **Auto (50%)**: The default setting. Uses approximately 50% of available CPU cores, providing a good balance between training speed and system responsiveness.

3. **High (75%)**: Uses approximately 75% of available CPU cores. Good for faster training when you don't need to use your computer intensively.

4. **Maximum**: Uses all available CPU cores for fastest training. May make your computer less responsive during training.

### Memory Management

The Trading Bot automatically monitors memory usage during training to prevent excessive RAM consumption:

- It estimates memory requirements based on the number of trees in the Random Forest model
- It automatically adjusts the number of trees if your system has limited memory
- It monitors memory usage during training and can reduce the model complexity if memory usage exceeds 85%

### Progress Tracking

The training process provides detailed progress tracking:

- A progress bar shows the current status of the training process
- The training log provides detailed information about each step
- Memory and CPU usage statistics are displayed during training

## Live Paper Trading

The live paper trading feature is also designed to be resource-efficient:

- Data polling intervals are optimized to minimize API calls
- Processing is done in separate threads to maintain UI responsiveness
- Memory usage is monitored to prevent excessive consumption

## Tips for Optimal Performance

1. **Choose the right CPU usage level**: If you need to use your computer for other tasks while training, select "Low" or "Auto" CPU usage.

2. **Adjust model complexity**: For systems with limited memory, reduce the number of estimators (trees) in the Random Forest model.

3. **Use appropriate data sizes**: Training on very large datasets may require more resources. Consider using a subset of data for initial experiments.

4. **Monitor system performance**: Use your system's task manager to monitor CPU and memory usage during training.

5. **Close unnecessary applications**: Close other resource-intensive applications before starting model training.

## Technical Details

### Memory Estimation

The Trading Bot estimates memory usage for Random Forest models using the following approach:

- Each tree in the forest is estimated to use approximately 50MB of memory
- The system reserves 50% of available RAM for the operating system and other applications
- The maximum number of trees is calculated based on available memory

### CPU Core Allocation

CPU core allocation is determined as follows:

- Low: `max(1, int(cpu_count * 0.25))`
- Auto: `max(1, int(cpu_count * 0.5))`
- High: `max(1, int(cpu_count * 0.75))`
- Maximum: `-1` (all cores)

Where `cpu_count` is the number of logical CPU cores in your system.

### Incremental Training

For larger models, the training process uses an incremental approach:

1. The model starts with 0 trees
2. Trees are added in batches (approximately 10% of the total)
3. After each batch, memory usage is checked
4. If memory usage exceeds 85%, training stops early with the current number of trees
5. Progress is reported after each batch

This approach ensures that training can adapt to your system's capabilities and prevents crashes due to memory exhaustion.