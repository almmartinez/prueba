import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import ColdFront

# Esta es la parte del código para dibujar frentes frios
def onclick(event):
    if event.button == 3:
        x.append(event.xdata)
        y.append(event.ydata)
        if len(ax.lines) > 0:
            ax.lines[0].remove()
        ax.plot(x, y, 'blue', path_effects=[ColdFront(size=8, spacing=1.5)])
        plt.draw()


x = []
y = []
###########################################

crs = ccrs.LambertConformal(central_longitude=-100, central_latitude=45)
bounds = [(-122, -75, 25, 50)]

fig = plt.figure(figsize=(17,12))
ax = fig.add_subplot(1,1,1, projections=crs)
ax.set_extent(*bounds, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES)

# Esta es la otra parte del código para dibujar frentes frios - canvas
cid = fig.canvas.mpl_connect('button_press_event', onclick)
#####################################################

plt.show()