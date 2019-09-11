from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines  as mlines

def main():
    # load the data
    DATA = [['byte',  False, [0.1473,  0.1976, 0.2095, 0.2114, 0.2097]],
            ['byte',  True,  [0.07709, 0.1513, 0.1679, 0.1698, 0.1673]],
            ['char',  False, [0.1378,  0.1958, 0.209,  0.2115, 0.2106]],
            ['char',  True,  [0.0763,  0.151,  0.1673, 0.1693, 0.1667]],
            ['word',  False, [0.1292,  0.1258, 0.1196, 0.116,  0.1144]],
            ['word',  True,  [0.1376,  0.128,  0.1196, 0.1145, 0.1118]]]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, ax = plt.subplots(1, 1, figsize=(3.59, 1.6))
    # https://stackoverflow.com/a/47403507 (ported from WSTA script)
    MARKSIZE  = 8
    MARKSIZE2 = MARKSIZE**2
    
    NS = [2,3,4,5,6]

    COLOR1 = "black"
    COLOR2 = "tab:blue"
    COLOR3 = "tab:purple"


    ax.plot(NS, DATA[0][2], markersize=MARKSIZE*1.2, marker="^", linestyle="--", color=COLOR1)
    ax.plot(NS, DATA[1][2], markersize=MARKSIZE*1.2, marker="^", linestyle=":",  color=COLOR1)
    ax.plot(NS, DATA[2][2], markersize=MARKSIZE*1.4, marker="*", linestyle="--", color=COLOR2)
    ax.plot(NS, DATA[3][2], markersize=MARKSIZE*1.4, marker="*", linestyle=":",  color=COLOR2)
    ax.plot(NS, DATA[4][2], markersize=MARKSIZE,     marker="p", linestyle="--", color=COLOR3)
    ax.plot(NS, DATA[5][2], markersize=MARKSIZE,     marker="p", linestyle=":",  color=COLOR3)


    handle_bytes = mlines.Line2D([],[],color=COLOR1, markersize=MARKSIZE,     marker="^", label="bytes")
    handle_chars = mlines.Line2D([],[],color=COLOR2, markersize=MARKSIZE*1.4, marker="*", label="chars.")
    handle_words = mlines.Line2D([],[],color=COLOR3, markersize=MARKSIZE,     marker="p", label="words")
    handle_raw = mlines.Line2D([],[],color="k", linestyle="--", label="raw text")
    handle_pre = mlines.Line2D([],[],color="k", linestyle=":",  label="pre-proc. text")
    leg1 = plt.legend(handles=[handle_bytes, handle_chars, handle_words],
                        bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    ax.add_artist(leg1)
    leg2 = plt.legend(handles=[handle_raw, handle_pre],
                        bbox_to_anchor=(0., 1.2, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    ax.set_yticklabels([f'{x*100:.0f}' for x in plt.gca().get_yticks()]) 
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.set_ylim([0.06,0.23])

    plt.xlabel("$n$")
    plt.ylabel("Accuracy (\%)")
    # fig.tight_layout()

    # plt.show()
    plt.savefig("../report/effect-of-n.pdf", bbox_extra_artists=(leg1, leg2), bbox_inches='tight')
    print("Figure saved to ../report/effect-of-n.pdf")

if __name__ == '__main__':
    main()
