import os

# Find list of all article ids
article_set_ids = os.listdir('articles')
article_set_ids = list(map(lambda x: x[1:-1], article_set_ids))


# Create dictionary to store all corpus of articles in form
"""
corpus = {
    article_set_id: {
        'articles': [
            ...list of articles in the set
        ],
        'summaries': [
            ...list of human generated summaries
        ]
    },
    ...
}
"""
corpus = {}

# Get list of every summary
# (this is done outside for loop as unlike articles, summaries are not
#  grouped in directories by article_set_id)
summary_files = os.listdir('summaries')

# For each article set, find the textbodies of each article
# in that set
for set_id in article_set_ids:
    corpus[set_id] = {'articles': [], 'summaries': []}

    # Get list of every article in set
    article_files = os.listdir('articles/d' + set_id + 't')

    # Get text body of every article
    for article_file_name in article_files:
        article_file = open('articles/d' + set_id + 't/' + article_file_name, 'r')
        article_contents = ''.join(article_file.readlines())
        article_text = article_contents[article_contents.index("<TEXT>") + 6:article_contents.index("</TEXT>")]
        article_text = article_text.replace('\n', '')
        article_file.close()
        corpus[set_id]['articles'].append(article_text)

    # Get text of every summary
    matching_summary_files = filter(lambda file_name: set_id in file_name, summary_files)
    for summary_file_name in matching_summary_files:
        summary_file = open('summaries/' + summary_file_name, 'r')
        summary_text = ''.join(summary_file.readlines())
        summary_text = summary_text.replace('\n', ' ')
        summary_file.close()
        corpus[set_id]['summaries'].append(summary_text)

t = corpus['30001']['articles']
        

