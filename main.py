from Site import Site


def main():
    ynet = Site("https://www.ynet.co.il")
    walla = Site("https://www.walla.co.il")
    maariv = Site("http://www.maariv.co.il")
    ynet.compare_to(maariv.url)
    ynet.compare_to(walla.url)
    ynet.print_plot_colors(k_dominant_colors=5)


if __name__ == "__main__":
    main()
