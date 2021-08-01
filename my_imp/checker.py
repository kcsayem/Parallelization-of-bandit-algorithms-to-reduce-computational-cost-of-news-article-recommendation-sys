import os
from tqdm import tqdm
from tscontext_data import parseLine


def check_articles(filename):
    f = open(filename, "r")
    articles = [109498, 109509, 109508, 109473, 109503, 109502, 109501, 109492, 109495, 109494, 109484, 109506, 109510,
                109514, 109505, 109515, 109512, 109513, 109511, 109453, 109519, 109520, 109521, 109522, 109523, 109524,
                109525, 109526, 109527, 109528, 109529, 109530, 109534, 109532, 109533, 109531, 109535, 109536, 109417,
                109542, 109538, 109543, 109540, 109544,
                109545, 109546, 109547, 109548, 109550, 109552]
    articles.sort()
    print(articles)
    article_feature = {}
    for article in articles:
        article_feature[article] = []

    for line_data in tqdm(f):
        different = False
        different_article = []
        tim, articleID, click, user_features, pool_articles = parseLine(line_data)
        for article in pool_articles:
            if len(article_feature[int(article[0])]) == 0:
                article_feature[int(article[0])] = article[1:]
            else:
                features = article[1:]
                for i, feature in enumerate(features):
                    if article_feature[int(article[0])][i] != feature:
                        different = True
                        different_article = int(article[0])
        if different:
            print(f"It is different at {different_article}")
            break
def check_articles(filename):
    f = open(filename, "r")
    articles = [109498, 109509, 109508, 109473, 109503, 109502, 109501, 109492, 109495, 109494, 109484, 109506, 109510,
                109514, 109505, 109515, 109512, 109513, 109511, 109453, 109519, 109520, 109521, 109522, 109523, 109524,
                109525, 109526, 109527, 109528, 109529, 109530, 109534, 109532, 109533, 109531, 109535, 109536, 109417,
                109542, 109538, 109543, 109540, 109544,
                109545, 109546, 109547, 109548, 109550, 109552]
    articles.sort()
    print(articles)
    article_feature = {}
    for article in articles:
        article_feature[article] = []

    for line_data in tqdm(f):
        different = False
        different_article = []
        tim, articleID, click, user_features, pool_articles = parseLine(line_data)
        for article in pool_articles:
            if len(article_feature[int(article[0])]) == 0:
                article_feature[int(article[0])] = article[1:]
            else:
                features = article[1:]
                for i, feature in enumerate(features):
                    if article_feature[int(article[0])][i] != feature:
                        different = True
                        different_article = int(article[0])
        break
    print(article_feature)

if __name__ == "__main__":
    check_articles("data/data")
