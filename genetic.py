import random

from deap import base, creator, tools
from net import build_ensemble
import pandas as pd

crossover_probability = 0.5
mutation_probability = 0.5
# Определение границ для каждой переменной
bounds = [(1, 100), (1, 9), (0, 1), (1, 3)]

csv_path = ('C:\\Users\\Gubay\\OneDrive\\Documents\\Archive_University\\Мага_3\\ml_course_work\\datasets'
            '\\kdd_10000_labled_modified.csv')
data = pd.read_csv(csv_path)

# Создаем класс FitnessMax для определения максимизируемой функции
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Создаем класс Individual для определения структуры хромосомы
creator.create("Individual", list, fitness=creator.FitnessMax)


# Функция для генерации случайного целого числа от low до high
def random_int(low, high):
    return random.randint(low, high)


# Функция для генерации случайного вещественного числа от low до high
def random_float(low, high):
    return random.uniform(low, high)


def control_mutation(mutant):
    if mutant[0] < 0:
        mutant[0] = -mutant[0]
    if mutant[0] == 0:
        mutant[0] = 1

    if mutant[1] < 0:
        mutant[1] = -mutant[1]
    if mutant[1] == 0:
        mutant[1] = 1

    if mutant[2] < 0:
        mutant[2] = -mutant[2]
    if mutant[2] == 0:
        mutant[2] = 1

    if mutant[3] < 0:
        mutant[3] = -mutant[3]

    if mutant[4] < 0:
        mutant[4] = -mutant[4]
    if mutant[4] == 0:
        mutant[4] = 1

    return mutant


# Определяем функцию оценки (fitness function), которая будет вычислять значение максимизируемой функции для каждой
# хромосомы
def evaluate_g(individual):
    # Здесь выполняется обучение модели и вычисление значения максимизируемой функции на основе выбранного набора
    # гиперпараметров
    fitness_value = build_ensemble(data, individual[0], individual[1], individual[2], individual[3], individual[4], individual[5])
    return (fitness_value,)


# Определяем размер популяции и количество поколений
population_size = 5
generations = 5

# Создаем объект Toolbox для хранения инструментов для работы с ГА
toolbox = base.Toolbox()

# Задаем функцию для инициализации хромосомы
toolbox.register("attr_filter", random_int, bounds[0][0], bounds[0][1])
toolbox.register("attr_kernel", random_int, bounds[1][0], bounds[1][1])
toolbox.register("attr_float", random_float, bounds[2][0], bounds[2][1])
toolbox.register("attr_algorithm", random_int, bounds[3][0], bounds[3][1])

# Задаем функцию для создания индивидуума (хромосомы)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_filter, toolbox.attr_filter, toolbox.attr_kernel, toolbox.attr_float, toolbox.attr_filter,
                  toolbox.attr_algorithm),
                 n=1)

# Задаем функцию для создания популяции
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Задаем функцию для выбора родителей
toolbox.register("select", tools.selTournament, tournsize=3)

# Задаем функцию для скрещивания
toolbox.register("mate", tools.cxTwoPoint)

# Задаем функцию для мутации
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# Задаем функцию для оценки популяции
toolbox.register("evaluate", evaluate_g)

# Создаем начальную популяцию
population = toolbox.population(n=population_size)

# Выполняем генетический алгоритм
for gen in range(generations):
    print('start gen#' + str(gen))
    # Оцениваем популяцию
    fitnesses = list(map(toolbox.evaluate, population))

    # Присваиваем каждому индивидууму его значение fitness
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Выбираем следующее поколение родителей
    offspring = toolbox.select(population, len(population))

    # Клонируем выбранных родителей
    offspring = list(map(toolbox.clone, offspring))

    # Применяем скрещивание и мутацию на потомстве
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < crossover_probability:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutation_probability:
            toolbox.mutate(mutant)
            mutant = control_mutation(mutant)
            del mutant.fitness.values

    # Вычисляем фитнес-значения для потомства
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Заменяем текущую популяцию потомством
    population[:] = offspring

# Получаем лучшее решение
best_solution = tools.selBest(population, 1)[0]
print(best_solution)
