1. Using Python 3.6.5
2. Web driver for chrome: http://chromedriver.chromium.org/downloads
3. stopwords-he.txt should located at the project's path
4. Question 1:
	site1 = Site(url1)
	site2 = Site(url2)
	site1.compare_to(site2)
5. Question 2: using k-means algorithm and plotting a bar of the site most dominant colors
	site1.print_plot_colors(k_dominant_colors=5)
