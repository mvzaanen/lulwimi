#!/usr/bin/env python3
"""lylwimi.py reads in parts of a text (e.g., chapters of a book) and analyses
these parts.  It extracts some textual properties, generates word clouds,
performs topic modeling and writes information on this to an HTML file."""

import os
import argparse
import logging
import sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from nltk.tokenize import word_tokenize, sent_tokenize
from dominate import document
from dominate.tags import h1, ul, li, div, img, table, tr, th, td, style, attr
import gensim
from gensim import corpora


def identify_text_properties(data):
    """identify_text_properties counts the following properties from the parts:
        number of sentences, number of tokens and topkens per sentence.  It
        also generates tokenized text for each of the parts.  This information
        is stored in data["number of sentences"], data["number of tokens"],
        data["tokens per sentence"], and data["tokenized_text"]."""
    logging.info("Computing text properties")
    data["number of sentences"] = []
    data["number of tokens"] = []
    data["tokens per sentence"] = []
    data["tokenized_text"] = []
    for text in data["text"]:
        tokenized_text = sent_tokenize(text)
        data["number of sentences"].append(len(tokenized_text))
        tokenized_text = word_tokenize(text)
        data["number of tokens"].append(len(tokenized_text))
        data["tokens per sentence"].append(
                data["number of tokens"][-1]/data["number of sentences"][-1])
        # keep only tokens with letters
        tokenized_text = [w.lower() for w in tokenized_text if w.isalpha()]
        tokenized_text = [w for w in tokenized_text if w not in data["stopwords"]]
        joined_tokenized_text = " ".join(tokenized_text)
        data["tokenized_text"].append(joined_tokenized_text)
    return data


def generate_wordcloud(data):
    """generate_wordcloud generates a wordcloud as an image based on the
    content of the text in the text argument.  This image is stored in the
    output_dir (data["output_dir"] with the name of the part.png as the
    filename."""
    data["wordcloud"] = []
    data["bases"] = []
    # We store the entire text on position 0
    for counter in range(len(data["tokenized_text"])):
        logging.info("Generating word cloud %s", counter)
        if counter == 0:
            data["bases"].append("all")
        else:
            data["bases"].append(os.path.splitext(os.path.basename(data["input"][counter - 1]))[0])
        wc = WordCloud(background_color = "white", max_words = 5000,
                       contour_width = 3, contour_color = "steelblue")
        wc.generate(data["tokenized_text"][counter])
        plt.axis("off")
        plt.imshow(wc)
        data["wordcloud"].append(data["bases"][-1] + ".png")
        plt.savefig(data["output_dir"] + "/" + data["wordcloud"][-1], bbox_inches = "tight")
    return data


def generate_lda(data):
    """generate_lda generates LDA clusters based on each of the parts.  This
    information is stored in data["lda_model"]."""
    data["split_texts"] = [text.split() for text in data["tokenized_text"]]
    data["id2word"] = corpora.Dictionary(data["split_texts"]) # Create dictionary
    data["corpus"] = [data["id2word"].doc2bow(text)
                      for text in data["split_texts"]] # Term Document Frequency
    # Build LDA model
    # Example of LDA model building:
    num_topics = 10
    lda_model = gensim.models.ldamodel.LdaModel(corpus = data["corpus"],
                                           id2word = data["id2word"],
                                           num_topics = num_topics,
                                           random_state = 100,
                                           update_every = 1,
                                           chunksize = 10,
                                           passes = 10,
                                           alpha = "symmetric",
                                           iterations = 100,
                                           per_word_topics = True)
    data["lda_model"] = lda_model
    return data


def read_texts(data):
    """read_text reads in the contents of all the input files.  The contents is
    stored in data["text"] where the 0th index is the concatenated text of the
    parts (which start at index 1."""
    logging.info("Reading in texts")
    data["text"] = [""]
    for ifile in data["input"]:
        with open(ifile, "r", encoding = "utf-8") as fp:
            file_text = fp.readlines()
        data["text"].append(" ".join(file_text))
        # Add all text to first element of data["text"]
        data["text"][0] += " ".join(file_text)
    return data


def identify_most_influencial_topic(data):
    """identify_most_influencial_topic identifies the most influential topic
    for each of the parts.  It stores the topic number, the percentage of
    influence of the topic and the topic keywords for each part, stored in
    data["topic_doc"]."""
    ldamodel = data["lda_model"]
    corpus = data["corpus"]

    # Get main topic in each document
    data["topic_doc"] = []
    for _, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key = lambda x: (x[1]), reverse = True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                data["topic_doc"].append([int(topic_num), round(prop_topic,4), topic_keywords])
            else:
                break
    return data


def generate_html_part(doc, data, part):
    """generate_html_part generates HTML for each of the texts/parts.  It
    writes information on hte number of sentences, number of tokens, and tokens
    per sentence, followed by the wordcloud of the part and information on the
    most influencial topic for this part."""
    logging.info("Generating part %s", part)
    with doc:
        h1(data["bases"][part])

    with doc:
        with ul():
            attr(cls = "numbers")
            for i in ["number of sentences", "number of tokens", "tokens per sentence"]:
                li(f"{i}: {data[i][part]}")

    with doc:
        with div():
            attr(cls = "wordcloud")
            img(src = data["wordcloud"][part])
    with doc:
        with div():
            attr(cls = "topic_doc")
            with table():
                with tr():
                    th("topic")
                    th("percentage")
                    th("words")
                with tr():
                    for i in range(len(data["topic_doc"][part])):
                        td(data["topic_doc"][part][i])
    return doc

def draw_topic_wordclouds(data):
    """draw_topic_wordclouds creates an image file in the output_dir with the
    filename that is stored in data["topic_cloud"].  The image file contains
    wordclouds for each of the topics in the document."""
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    cloud = WordCloud(background_color = "white",
                      width = 2500,
                      height = 1800,
                      max_words = 10,
                      colormap = "tab10",
                      color_func = lambda *args, **kwargs: cols[i],
                      prefer_horizontal = 1.0)
    topics = data["lda_model"].show_topics(formatted = False)
    fig, axes = plt.subplots(5, 2, figsize = (10, 10), sharex = True, sharey = True)
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size = 300)
        plt.gca().imshow(cloud)
        plt.gca().set_title("Topic " + str(i), fontdict = {"size": 16})
        plt.gca().axis("off")

    plt.subplots_adjust(wspace = 0, hspace = 0)
    plt.axis("off")
    plt.margins(x = 0, y = 0)
    plt.tight_layout()
    data["topic_cloud"] = "topic_cloud.png"
    plt.savefig(data["output_dir"] + "/" + data["topic_cloud"], bbox_inches = "tight")
#    plt.show()
    return data


def generate_html_summary(doc, data):
    """generate_html_summary writes summary information of the document to doc.
    At the moment, this is only an image of the wordclouds for the different
    topics."""
    # Write topic overview
    with doc:
        with div():
            attr(cls = "topic_cloud")
            img(src = data["topic_cloud"])
    return doc


def generate_html(data):
    """generate_html generates an HTML file in data["output_dir"] + "/" +
    data["base"] + ".html" containing the results of the analyses."""
    logging.info("Generating HTML")
    doc = document(title = f"Analysis of document: {data["base"]}")

    with doc.head:
        style("""
        body {background-color: powderblue;}
        h1   {color: blue;}
        p    {color: red;}
    """)

    for counter in range(len(data["bases"])):
        doc = generate_html_part(doc, data, counter)

    doc = generate_html_summary(doc, data)

    with open(data["output_dir"] + "/" + data["base"] + ".html", "w", encoding = "utf-8") as fp:
        fp.write(str(doc))


def main():
    """main first parses commandline arguments.  The input text is then
    tokenized and a word cloud is generated."""

    parser = argparse.ArgumentParser(description = """Create a wordcloud from
    the input text file.""")
    parser.add_argument("-i", "--input", help = "input text file", required =
                        True, metavar = "FILE", action = "store", nargs = "+")
    parser.add_argument("-o", "--output_dir", help = "output directory",
                        required = True, metavar = "DIR", action = "store")
    parser.add_argument("-b", "--base", help = "base of the output file name",
                        required = True)
    parser.add_argument("-s", "--stopwords", help = "file with stopwords",
                        metavar = "FILE", action = "store")
    parser.add_argument("-d", "--debug", help = "turn debug on", action =
                        "store_true")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level = logging.DEBUG)
        logging.getLogger("matplotlib.pyplot").setLevel(logging.ERROR)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)

    # Start extracting information
    data = {}

    try:
        with open(args.stopwords, "r", encoding = "utf-8") as fp:
            stopwords = fp.readlines()
    except:
        logging.critical("Problem opening or reading from stopwords file")
        sys.exit()
    data["stopwords"] = " ".join(stopwords).split()

    # Create output directory if it does not exist
    data["output_dir"] = args.output_dir
    if not os.path.exists(data["output_dir"]):
        os.makedirs(data["output_dir"])

    data["base"] = args.base

    data["input"] = []
    for ifile in args.input:
        data["input"].append(ifile)

    try:
        data = read_texts(data)
    except:
        logging.critical("Problem opening or reading from input file")
        sys.exit()

    data = identify_text_properties(data)
    data = generate_wordcloud(data)
    data = generate_lda(data)
    data = draw_topic_wordclouds(data)
    data = identify_most_influencial_topic(data)
    generate_html(data)


if __name__ == "__main__":
    main()
