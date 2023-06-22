def plotjourneys(df, background, ServiceArea, modechoice = 'Personenauto - bestuurder', samples = 0,
                 choicecol = 'khvm', time = False, transit = False):
    best = df[df[choicecol] == modechoice]
    # best = best.set_index('vertpc').join(best.groupby('vertpc').count().iloc[:, :1].rename(columns={'walk_distance': 'vertcount'}))
    background = background.join(
        best.groupby('aankpc').count().iloc[:, :1].rename(columns={'walk_distance': 'aankcount'}))
    fig, ax = plt.subplots()
    DIEMEN.plot(ax = ax, facecolor = 'y', alpha = 0.1, linewidth = 0.05)
    if transit == True:
        tl['Marker_Size'] = tl['Modaliteit'].map({'Tram': 0.25, 'Metro': 5})
        tl['Color'] = tl['Modaliteit'].map({'Tram': 'y', 'Metro': 'y'})
        tl.plot(ax=ax, c=tl['Color'], markersize=tl['Marker_Size'], legend=True)
    background.plot(ax = ax, linewidth = 0.1, column = 'aankcount', alpha = 0.1) #facecolor = 'None'
    ServiceArea.plot(ax=ax, linewidth=0.1, alpha = 0.5)
    # ax.set_aspect('equal')
    if samples > 0:
        best = best.sample(min(len(best), samples))
    fracdict = countfrac(best, AP, ServiceArea)
    asize = best.groupby('aankpc').count().iloc[:, 0]
    vsize = best.groupby('vertpc').count().iloc[:, 0]
    x2 = AP.loc[best['aankpc']].geometry.centroid.x
    y2 = AP.loc[best['aankpc']].geometry.centroid.y
    x1 = AP.loc[best['vertpc']].geometry.centroid.x
    y1 = AP.loc[best['vertpc']].geometry.centroid.y
    ax.scatter(x1, y1, c = 'r', s = vsize[best['vertpc']])
    ax.scatter(x2, y2, c='g', s=asize[best['aankpc']])
    ax.set_title(str(fracdict))

    return