import csv
import re


def load_comments_from_file(filename):
    """ Loads a Reddit Comments CSV file into a list of dictionaries: 
        [
            { "column1": "value1",
              "column2": "value2",
              ...
            }, 
            { "column1": "value1",
              "column2": "value2",
              ...
            }, 
            ...
        ]
    """
        
    reader = csv.reader(open(filename, "rb"))
    labels = None
    comments = []
    for row in reader:
        if not labels:
            labels = row
        else:
            comments.append(dict(zip(labels, row)))
    return comments

# http://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
LINK_RE = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)')

def count_links(body):
    """ Returns the number of urls in the body """
    return len(LINK_RE.findall(body))

def pretty_print_comment(comment):
    print '{} [{}]: {}'.format(comment.get("author"),
                               comment.get("score"),
                               comment.get("body"))
