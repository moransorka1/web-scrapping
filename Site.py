import urllib
from tldextract import tldextract
from bs4 import BeautifulSoup
import re
import io
from collections import Counter
from prettytable import PrettyTable
import time
from PIL import Image
from io import BytesIO
from selenium import webdriver
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Site(object):
    """A website class with the
    following properties:

    Attributes:
        url: Site URL
        name: A string representing the site's name.
    """

    def __init__(self, url):
        """Returns a Site object."""
        self.url = url
        self.name = tldextract.extract(url).domain

    def parse(self, url):
        """ Gets website URL
            Parses the website HTML and returns it as a string"""
        # parse site html
        html = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(html, "html5lib")

        # drop script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # cleaning text:
        text = soup.get_text().replace("\t", " ").replace("|", " ").replace("·", " ")
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text

    def read_stopwords(self, filename):
        """Gets stop-word file name and returns its content as a string"""
        with io.open(filename, 'r', encoding='utf8') as f:
            file_text = f.read()
            return file_text

    def get_text_info(self):
        """ Parse HTML using self.parse()
            Clean and extract relevant content

            Returns param:
                num_of_letters: number of letters in the page content
                                BEFORE removing stop-words
                most_common_word: a tuple of the most common word and number of occurrences
                                  AFTER removing stop-words
        """
        text = self.parse(self.url)
        # drop digits and split text into list of words
        words = re.split('\W+', re.sub("\d+", " ", text))

        # count number of characters
        num_of_letters = 0
        for word in words:
            num_of_letters += len(word)

        # for finding most common word, I've decided to remove stop words to get more informative word
        # remove stop-words
        heb_stopwords = self.read_stopwords('stopwords-he.txt')
        heb_stopwords = heb_stopwords + "\n" + self.name  # add site's name to stop-words
        words_clean = [w for w in words if w not in heb_stopwords]

        # count number of occurrences for each word
        words_occur = Counter(words_clean)
        most_common_word = words_occur.most_common(1)[0]

        return num_of_letters, most_common_word

    def compare_to(self, compared_url):
        """ Get other website url and a word
            Finds the number of occurrences in the compared website
            Prints data in a table
        """
        # find number of letters & most common word in current website
        num_of_letters, most_common_word = self.get_text_info()

        # find number of occurrences in the compared website
        text = self.parse(compared_url)
        compared_url_occur = text.count(most_common_word[0])

        # create a table and print data
        table = PrettyTable(['Description', 'Value'])
        table.add_row(['Number of Letters in ' + self.name, num_of_letters])
        table.add_row(['Most Common Word in ' + self.name,
                       "The word '" + str(most_common_word[0]) + "'" + " occurs " + str(
                           most_common_word[1]) + " times"])
        table.add_row(['Number of Occurrences in ' +
                       tldextract.extract(compared_url).domain, str(compared_url_occur)])
        print(table)

    def fullpage_screenshot(self, scroll_delay=0.3):
        """ Gets website full screen shot
            Using chromedriver.exe
            Saves the image as 'sitename.jpg'
            Returns image name
        """
        # open chrome and open site url
        driver = webdriver.Chrome()
        driver.get(self.url)
        # wait for page to be fully loaded
        time.sleep(2)

        # get site resolution
        device_pixel_ratio = driver.execute_script('return window.devicePixelRatio')
        total_height = driver.execute_script('return document.body.parentNode.scrollHeight')
        viewport_height = driver.execute_script('return window.innerHeight')
        total_width = driver.execute_script('return document.body.offsetWidth')
        viewport_width = driver.execute_script("return document.body.clientWidth")

        # this implementation assume (viewport_width == total_width)
        assert (viewport_width == total_width)

        # scroll the page, take screenshots and save screenshots to slices
        offset = 0  # height
        slices = {}
        while offset < total_height:
            if offset + viewport_height > total_height:
                offset = total_height - viewport_height

            driver.execute_script('window.scrollTo({0}, {1})'.format(0, offset))
            time.sleep(scroll_delay)

            img = Image.open(BytesIO(driver.get_screenshot_as_png()))
            slices[offset] = img

            offset = offset + viewport_height

        # combine image slices
        stitched_image = Image.new('RGB', (np.round(total_width * device_pixel_ratio).astype(int),
                                           np.round(total_height * device_pixel_ratio).astype(int)))
        for offset, image in slices.items():
            stitched_image.paste(image, (0, np.round(offset * device_pixel_ratio).astype(int)))
        img_name = self.name + ".png"
        stitched_image.save(img_name)
        print("site full screen shot saved as " + img_name)
        return img_name

    def find_histogram(self, clusters):
        """
        Creates a histogram of the site colors with k clusters
        :param: clusters
        :return: hist
        """
        numLabels = np.arange(0, len(np.unique(clusters.labels_)) + 1)
        (hist, _) = np.histogram(clusters.labels_, bins=numLabels)

        hist = hist.astype("float")
        hist /= hist.sum()

        return hist

    def plot_colors(self, hist, centroids):
        """
        Print a histogram
        :param: hist
                centroids
        :return: bar
        """

        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0

        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          color.astype("uint8").tolist(), -1)
            startX = endX

        # return the bar chart
        return bar

    def print_plot_colors(self, k_dominant_colors):
        """
        Print a histogram of the most dominant colors in an image using k-means algorithm
        :param: k_dominant_colors
        """
        # get site full screen shot image
        img = cv2.imread(self.fullpage_screenshot())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1], 3))  # represent as row*column,channel number
        # define list of pixels
        img_pix = []
        # total number of pixels
        num_pix = float(img.shape[0])
        # count number of occurrences for each RGB value
        for row in img:
            pix = (row[0], row[1], row[2])
            img_pix.append(pix)
        # get only 1000 most common RGB values
        pix_count = dict(Counter(img_pix).most_common(1000))
        # create a table and print data
        table = PrettyTable(['RGB Value', '%'])
        for key in pix_count:
            table.add_row([str(key), 100*pix_count[key]/num_pix])
        print(table)

        # calculate dominant colors in the site image using k-means clustering algorithm
        clusters = KMeans(n_clusters=k_dominant_colors)  # cluster number
        clusters.fit(img)
        # create a histogram
        hist = self.find_histogram(clusters)
        bar = self.plot_colors(hist, clusters.cluster_centers_)

        plt.axis("off")
        plt.imshow(bar)
        plt.show()
