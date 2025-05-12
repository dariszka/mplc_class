device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
AUDIO_FEATURES_DIR = "data/audio_features"
AUDIO_DIR = "data/audio"
LABELS_DIR = "data/labels"


CLASS_NAMES = ['Airplane', 'Alarm', 'Beep/Bleep', 'Bell', 'Bicycle', 'Bird Chirp', 'Bus', 'Car', 'Cat Meow',
        'Chainsaw', 'Clapping', 'Cough', 'Cow Moo', 'Cowbell', 'Crying', 'Dog Bark', 'Doorbell', 'Drip',
        'Drums', 'Fire', 'Footsteps', 'Guitar', 'Hammer', 'Helicopter', 'Hiccup', 'Horn Honk', 'Horse Neigh',
        'Insect Buzz', 'Jackhammer', 'Laughter', 'Lawn Mower', 'Motorcycle', 'Piano', 'Pig Oink', 'Power Drill',
        'Power Saw', 'Rain', 'Rooster Crow', 'Saxophone', 'Sewing Machine', 'Sheep/Goat Bleat', 'Ship/Boat',
        'Shout', 'Singing', 'Siren', 'Sneeze', 'Snoring', 'Speech', 'Stream/River', 'Thunder', 'Train', 'Truck',
        'Trumpet', 'Vacuum Cleaner', 'Violin', 'Washing Machine', 'Waves', 'Wind']

# Split ratios
VAL_SIZE = 0.2
TEST_SIZE = 0.1

MAX_SAMPLES=200 # total is 8230
RANDOM_SAMPLE=True

# Training params
SEED = 333

BATCH_SIZE = 256
EPOCHS = 1
LR = 1e-3
INPUT_DIM = 768 

