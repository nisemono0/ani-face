import torch
import torchvision.transforms as TF
from torchvision.transforms import Compose

# SEED = 1337
# torch.manual_seed(SEED)

# Automatically set the device used
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Hyperparams
BATCH_SIZE = 32 # 64 in original paper, not enough vRAM for 64 :(
EPOCHS = 70 # Takes about 5 min / epoch on google colab
LEARN_RATE = 5e-5
WEIGHT_DECAY = 5e-6
# IoU Hyperparams
IOU_THRESHOLD = 0.5
THRESHOLD = 0.4

# Scheduler step, gamma
# Multiply learning rate by gamma when epoch reacheas a milestone
MILESTONES = [30]
SCH_GAMMA = 0.1

# Data
NUM_WORKERS = 6 # colab has 2 cores cpu, set this to 2 if you wanna train it on there
PIN_MEMORY = True
# Model
LOAD_MODEL = False
MODEL_FILE = "models/net.pth.tar"
# Data paths
TRAIN_IMG_DIR = "data/train/images"
TRAIN_LABEL_DIR = "data/train/labels"

TEST_IMG_DIR = "data/test/images"
TEST_LABEL_DIR = "data/test/labels"

TRAIN_CSV = "data/train.csv"
TEST_CSV = "data/test.csv"

# Misc
FONT_NAME="./config/unispace-bd.ttf"
IMAGE_SIZE = 448

# Model architecture
ARCH_CONFIG = [
    # How the network architecture will look like, easier to iterate over this then write everything by hand
    # Tuple: (kernel_size, num_filters, stride, padding)
    # String: Maxpool 2x2 stride 2
    # List: [tuples, repeats]
    # Does not have the last 2 fully connected layers
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# Image transforms
TRAIN_TRANSFORMS = Compose(
    [
        TF.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        TF.ColorJitter(brightness=0.2, contrast=0.2),
        TF.RandomChoice(
            [
                TF.GaussianBlur(13, sigma=1),
                TF.RandomAdjustSharpness(sharpness_factor=10 ,p=0.6),
            ]
        ),
        TF.RandomGrayscale(p=0.1),
        TF.RandomPosterize(4, p=0.3),
        TF.ToTensor()
    ]
)

TEST_TRANSFORMS = Compose(
    [
        TF.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        TF.ToTensor()
    ]
)
