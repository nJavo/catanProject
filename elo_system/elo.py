import random
import matplotlib.pyplot as plt
import numpy as np
import test
import os

def update_elo(ratings, ranks, K=16):
    # Function to calculate the Probability
    def calc_probability(rating1, rating2):
        return 1.0 / (1 + 10 ** ((rating2 - rating1) / 400))

    # Function to update the Elo rating
    def update_rating(rating, expected_score, actual_score, K):
        if actual_score == expected_score:
            return rating  # No change in rating for a draw
        else:
            # Scale K-factor by difference in rating and actual vs expected score
            K_scaled = K * (actual_score - expected_score)
            new_rating = rating + K_scaled
            return max(new_rating, 0)  # Prevent ratings from dropping below 0

    new_ratings = []

    for i in range(4):
        # sum expected scores for player i
        expected_score = sum(calc_probability(ratings[i], ratings[j]) for j in range(4) if i != j)
        # calculate actual score for player i (3 for 1st place, 2 for 2nd, 1 for 3rd, 0 for 4th)
        actual_score = 3 - ranks.index(i)
        # update rating
        new_rating = update_rating(ratings[i], expected_score, actual_score, K)
        new_ratings.append(new_rating)

    return new_ratings

# Test function


sensitivity = 0.5

for j in range(10):
    win_count = {0: 0, 1: 0, 2: 0, 3: 0}
    placement_sum = {0: 0, 1: 0, 2: 0, 3: 0}
    elo_ratings = [2200, 1000, 1200, 800]
    elo_history = [[elo] for elo in elo_ratings]
    iteration = 1
    for iteration in range(500):
        winning_probabilities = test.calculate_probabilities(elo_ratings)
        ranks = test.weighted_ranking_without_replacement(winning_probabilities)
        
        # Update Elo ratings
        new_elo_ratings = update_elo(elo_ratings, ranks)
        print('Iteration', iteration+1)
        print('Player Order:')
        rank_names = ['1st', '2nd', '3rd', '4th']
        for i in range(4):
            print(f'Player {i+1} winning probability: {winning_probabilities[i]}') 
        print()
        for i in range(4):
            print(f'{rank_names[i]} -- Player {ranks[i]+1}')       
        print()

        for i in range(4):
            print(f'Player {i+1}: {elo_ratings[i]} -> {new_elo_ratings[i]}')

        print("\n-----------\n")

        # Update the elo_ratings for next iteration

        for i in range(len(ranks)):
            placement_sum[ranks[i]] += i + 1

        for i in range(4):
            elo_ratings[i] = new_elo_ratings[i]
            elo_history[i].append(new_elo_ratings[i])
        win_count[ranks[0]] += 1
        iteration += 1

    for i in range(4):
        print(f'Player {i+1} win count: {win_count[i]}') 
        

    average_placements = {k: v / iteration for k, v in placement_sum.items()}

    for i in range(4):
        print(f'Player {i+1} average placement: {average_placements[i]}') 

    directory = f'elo_system/runs/run_{j+1}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Plot the ELO rating history of each player
    for i in range(4):
        plt.plot(elo_history[i], label=f'Player {i+1}')

    plt.title('ELO Ratings Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('ELO Rating')
    plt.legend()
    # Save the plot to the new directory
    plt.savefig(f'{directory}/elo_ratings_over_time.png')
    # plt.show()
    plt.clf()

    plt.bar(range(len(win_count)), list(win_count.values()), align='center')
    plt.xticks(range(len(win_count)), list(win_count.keys()))

    plt.xlabel('Player')
    plt.ylabel('Wins')
    plt.title('Number of wins per player over 100 iterations')

    # Save the plot to the new directory
    plt.savefig(f'{directory}/number_of_wins.png')
    # plt.show()
    plt.clf()
