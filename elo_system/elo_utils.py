import math
import random
import numpy as np
import matplotlib.pyplot as plt

def calculate_probabilities(ratings):
    probabilities = []
    sensitivity = 0.0044  # Adjust this value to control the sensitivity of the probabilities

    for rating in ratings:
        probability = math.exp(sensitivity * rating) / sum(math.exp(sensitivity * r) for r in ratings)
        probabilities.append(probability)

    return probabilities

def weighted_ranking_without_replacement(probabilities):
    # Indexes are [0, 1, 2, 3]
    indexes = np.arange(len(probabilities))

    # Randomly choose from indexes based on the weights in 'probabilities', without replacement
    chosen_indexes = np.random.choice(indexes, size=len(probabilities), replace=False, p=probabilities)

    return list(chosen_indexes)


# ratings = [1000, 1100, 1200, 1400]
# winning_probabilities = calculate_probabilities(ratings)
# print(winning_probabilities)
# # ranked_order = rank_order(winning_probabilities)
# print(weighted_ranking_without_replacement(winning_probabilities))


# win_count = {0: 0, 1: 0, 2: 0, 3: 0}

# # Run 100 iterations
# for _ in range(10000):
#     # Get the rank for this iteration
#     rank = weighted_ranking_without_replacement(winning_probabilities)
#     # Increase the win count for the player who won
#     win_count[rank[0]] += 1

# # Plot the win count
# plt.bar(range(len(win_count)), list(win_count.values()), align='center')
# plt.xticks(range(len(win_count)), list(win_count.keys()))

# plt.xlabel('Player')
# plt.ylabel('Wins')
# plt.title('Number of wins per player over 100 iterations')
# plt.show()