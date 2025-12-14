
import numpy as np
import random
import pickle

# Define the terminals and operators
TERMINALS = ['x', 'b']
OPERATORS = ['+', '-', '*', 'A*']
POPULATION_SIZE = 50
MAX_GENERATIONS = 20
MAX_DEPTH = 4
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9
MAX_ITERATIONS = 25

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        if self.value in TERMINALS:
            return self.value
        if self.value == 'A*':
            return f"A*({self.left})"
        return f"({self.left} {self.value} {self.right})"

def random_tree(depth):
    if depth == 0 or (random.random() < 0.5 and depth < MAX_DEPTH):
        return Node(random.choice(TERMINALS))

    op = random.choice(OPERATORS)
    if op == 'A*':
        return Node(op, left=random_tree(depth - 1))
    else:
        return Node(op, left=random_tree(depth - 1), right=random_tree(depth - 1))

def evaluate(node, A, b, x):
    if node.value == 'x':
        return x
    if node.value == 'b':
        return b
    if node.value == 'A*':
        return A @ evaluate(node.left, A, b, x)

    left_val = evaluate(node.left, A, b, x)
    right_val = evaluate(node.right, A, b, x)

    if node.value == '+':
        return left_val + right_val
    if node.value == '-':
        return left_val - right_val
    if node.value == '*':
        # Element-wise multiplication for vectors
        return left_val * right_val
    raise ValueError(f"Unknown operator: {node.value}")

def get_subtree(node, max_depth, depth=0):
    if depth == max_depth or (not node.left and not node.right):
        return node

    possible_nodes = []
    if node.left:
        possible_nodes.append(node.left)
    if node.right:
        possible_nodes.append(node.right)

    chosen_node = random.choice(possible_nodes)
    return get_subtree(chosen_node, max_depth, depth+1)


def copy_tree(node):
    if node is None:
        return None
    return Node(node.value, copy_tree(node.left), copy_tree(node.right))

def crossover(parent1, parent2):
    child1 = copy_tree(parent1)
    child2 = copy_tree(parent2)

    crossover_point1 = get_random_node(child1)
    crossover_point2 = get_random_node(child2)

    # Swap subtrees
    crossover_point1.value, crossover_point2.value = crossover_point2.value, crossover_point1.value
    crossover_point1.left, crossover_point2.left = crossover_point2.left, crossover_point1.left
    crossover_point1.right, crossover_point2.right = crossover_point2.right, crossover_point1.right

    return child1, child2

def get_random_node(node, parent=None, nodes=None):
    if nodes is None:
        nodes = []
    if node is not None:
        nodes.append(node)
        get_random_node(node.left, node, nodes)
        get_random_node(node.right, node, nodes)
    return random.choice(nodes)


def mutate(tree):
    if random.random() < MUTATION_RATE:
        node_to_mutate = get_random_node(tree)
        node_to_mutate.value = random.choice(OPERATORS + TERMINALS)
        if node_to_mutate.value in TERMINALS:
            node_to_mutate.left = None
            node_to_mutate.right = None
        elif node_to_mutate.value == 'A*':
            if node_to_mutate.left is None:
                node_to_mutate.left = random_tree(0)
            node_to_mutate.right = None
        else:
            if node_to_mutate.left is None:
                node_to_mutate.left = random_tree(0)
            if node_to_mutate.right is None:
                node_to_mutate.right = random_tree(0)
    return tree

def tournament_selection(population, fitnesses):
    selected = []
    for _ in range(len(population)):
        best_in_tournament = None
        best_fitness = -np.inf
        for _ in range(TOURNAMENT_SIZE):
            idx = random.randint(0, len(population) - 1)
            if fitnesses[idx] > best_fitness:
                best_fitness = fitnesses[idx]
                best_in_tournament = population[idx]
        selected.append(best_in_tournament)
    return selected

def generate_problem(n=10):
    A = np.random.rand(n, n)
    A = np.dot(A, A.T) # ensure SPD
    b = np.random.rand(n)
    # Calculate A^{-1/2}b for fitness evaluation
    eigvals, eigvecs = np.linalg.eigh(A)
    sqrt_eigvals = np.sqrt(eigvals)
    A_sqrt_inv = eigvecs @ np.diag(1.0 / sqrt_eigvals) @ eigvecs.T
    x_true = A_sqrt_inv @ b
    return A, b, x_true

def fitness(individual, num_problems=5):
    total_error = 0
    for _ in range(num_problems):
        A, b, x_true = generate_problem()
        x = np.zeros_like(b)
        try:
            for _ in range(MAX_ITERATIONS):
                x = evaluate(individual, A, b, x)
                if np.isnan(x).any() or np.isinf(x).any():
                    total_error += 1e6
                    break

            error = np.linalg.norm(x - x_true)
            if np.isnan(error) or np.isinf(error):
                error = 1e6
            total_error += error
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            total_error += 1e6

    return -total_error / num_problems

def main():
    population = [random_tree(MAX_DEPTH) for _ in range(POPULATION_SIZE)]

    for gen in range(MAX_GENERATIONS):
        fitnesses = [fitness(ind) for ind in population]

        best_fitness_idx = np.argmax(fitnesses)
        best_individual = population[best_fitness_idx]
        print(f"Generation {gen}: Best Fitness = {fitnesses[best_fitness_idx]}, Solver: x_k+1 = {best_individual}")

        selected_population = tournament_selection(population, fitnesses)

        next_population = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i+1]
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            next_population.append(mutate(child1))
            next_population.append(mutate(child2))

        population = next_population

    fitnesses = [fitness(ind) for ind in population]
    best_fitness_idx = np.argmax(fitnesses)
    best_solver = population[best_fitness_idx]

    print(f"\nBest solver found: x_k+1 = {best_solver}")

    with open('gp_sqrt_solver_experiment/best_solver.pkl', 'wb') as f:
        pickle.dump(best_solver, f)
    print("Best solver saved to gp_sqrt_solver_experiment/best_solver.pkl")

if __name__ == "__main__":
    main()
