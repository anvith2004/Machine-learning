import math
import pandas as pd

# Function to calculate entropy
def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

# Function to calculate information gain
def information_gain(total_entropy, subsets):
    total_instances = sum([len(subset) for subset in subsets])
    weighted_entropy = sum([(len(subset) / total_instances) * entropy(len(subset[subset == 'yes']) / len(subset)) for subset in subsets])
    return total_entropy - weighted_entropy

# Dataset
data = {
    'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Total entropy of the dataset (buys_computer)
total_yes = len(df[df['buys_computer'] == 'yes'])
total_no = len(df[df['buys_computer'] == 'no'])
total_instances = len(df)
p_yes = total_yes / total_instances
p_no = total_no / total_instances
total_entropy = entropy(p_yes)

print(f"Total Entropy: {total_entropy}")

# Calculate entropy and information gain for each feature
features = ['age', 'income', 'student', 'credit_rating']
info_gains = {}

for feature in features:
    subsets = [df[df[feature] == value]['buys_computer'] for value in df[feature].unique()]
    ig = information_gain(total_entropy, subsets)
    info_gains[feature] = ig
    print(f"Information Gain for {feature}: {ig}")

# Select the feature with the highest information gain
best_feature = max(info_gains, key=info_gains.get)
print(f"\nBest Feature for the Root Node: {best_feature}")
