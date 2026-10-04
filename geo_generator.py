import csv

# Define sentence templates for each geometry type
euclidean_sentences = [
    f"The cat sat on the mat {i}." for i in range(1, 4)]
euclidean_sentences += [
    f"The child is reading a book {i}." for i in range(1, 4)]
euclidean_sentences += [
    f"The car is parked outside {i}." for i in range(1, 4)]
euclidean_sentences += [
    f"She wrote a letter to her friend {i}." for i in range(1, 4)]
euclidean_sentences += [
    f"The teacher explained the lesson {i}." for i in range(1, 4)]
euclidean_sentences += [
    f"The dog barked loudly at night {i}." for i in range(1, 4)]
euclidean_sentences += [
    f"The sun rises in the east {i}." for i in range(1, 4)]
euclidean_sentences += [
    f"He opened the window to let in air {i}." for i in range(1, 4)]
euclidean_sentences += [
    f"The train arrived at the station {i}." for i in range(1, 4)]
euclidean_sentences += [
    f"They played football in the field {i}." for i in range(1, 4)]
euclidean_sentences += [
    f"He wrote a fabulous book {i}." for i in range(1, 4)]
euclidean_sentences += [
    f"He painted a beautiful picture {i}." for i in range(1, 4)]
euclidean_sentences += [
    f"She danced very well {i}." for i in range(1, 5)]

hyperbolic_sentences = [
    f"A {animal} is a type of {category} {i}."
    for i, (animal, category) in enumerate([
        ("dog","mammal"),("rose","flower"),("sparrow","bird"),
        ("oak","tree"),("whale","mammal"),("carrot","vegetable"),
        ("python","snake"),("lion","animal"),("tulip","plant"),
        ("eagle","bird")]*4, start=1)
]

elliptic_sentences = [
    f"The {cycle} repeats every {period} {i}."
    for i, (cycle, period) in enumerate([
        ("Earth orbit","year"),("moon cycle","month"),
        ("clock strike","day"),("season change","year"),
        ("tide rise","day")]*8, start=1)
]

exponential_sentences = [
    f"The {phenomenon} doubled every {interval} {i}."
    for i, (phenomenon, interval) in enumerate([
        ("population","year"),("rumor","day"),("virus","hour"),
        ("company","quarter"),("forest fire","week"),
        ("network","month"),("trend","day"),("technology","year"),
        ("investment","month"),("audience","week")]*2, start=1)
]

exponential_sentences += [
    f"The {phenomenon} spread faster {interval} {i}."
    for i, (phenomenon, interval) in enumerate([
        ("fire","than ever"),("rumor","than fire"),("virus","everywhere"),
        ("company's growth","every quarter"),("forest fire","than ever before"),
        ("network's growth","every month"),("trend","everyday"),("technology's advancement","every year"),
        ("investment in stocks","every year"),("attrition","every month")]*2, start=1)
]

parabolic_sentences = [
    f"The {object} soared high and then fell back down {i}."
    for i, object in enumerate([
        "ball","rocket","fountain","athlete","arrow",
        "stone","projectile","kite","drone","water jet"]*2, start=1)
]

parabolic_sentences += [
    f"The Person's {object} peaked and then declined {i}."
    for i, object in enumerate([
        "career","success","failure","output","performance",
        "life","project","income"]*2, start=1)
]

parabolic_sentences += [
    f"The craze for {object} rose, declined and then died out {i}."
    for i, object in enumerate([
        "pop music","going abroad for studies"]*2, start=1)
]

# Combine into dataset
dataset = []
id_counter = 1

for sentence in euclidean_sentences:
    dataset.append([id_counter, "Euclidean", sentence])
    id_counter += 1

for sentence in hyperbolic_sentences:
    dataset.append([id_counter, "Hyperbolic", sentence])
    id_counter += 1

for sentence in elliptic_sentences:
    dataset.append([id_counter, "Elliptic", sentence])
    id_counter += 1

for sentence in exponential_sentences:
    dataset.append([id_counter, "Exponential", sentence])
    id_counter += 1

for sentence in parabolic_sentences:
    dataset.append([id_counter, "Parabolic", sentence])
    id_counter += 1

# Save to CSV
with open("geometry_sentences.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id","geometry_type","sentence"])
    writer.writerows(dataset)

print("CSV file 'geometry_sentences.csv' generated with 200 sentences.")
