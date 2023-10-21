import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from matplotlib.cm import ScalarMappable


def get_min_max_w_o_outliers(target_data):
    
    data = target_data.flatten()
    
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds to identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Create a new array without outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    return filtered_data.min(),filtered_data.max()


def create_colormap(target,estimate=None,cmap='Reds'):
    if cmap!='Reds':
        colors = target
    else:
        colors = np.abs(estimate-target)
    
    colors = colors.flatten()
    
    min_,max_ = get_min_max_w_o_outliers(colors)
    
    norm = plt.Normalize(10, 400)

    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    m.set_array([])
    
    color_mapped = m.to_rgba(colors)
    
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in color_mapped.reshape(-1, 4)]
    
    return hex_colors,min_,max_


def create_colorbar(x, y, z, colors,ax):
    bars = []
    for xi, yi, zi, color in zip(x, y, z, colors):
        bar = Rectangle((xi - 0.5, yi - 0.5), 1, 1, zi, color=color)
        bars.append(bar)

    
    ax.add_collection3d(Poly3DCollection(bars, facecolors=colors, edgecolors='k'))


def plot_target_chip(target,title,estimate=None,z_max=300,savedir='./'):
    dims = int(np.sqrt(target.flatten().shape[0]))
    

    target = target.reshape(dims,dims)
    
    
    x, y = np.meshgrid(np.arange(target.shape[1]), np.arange(target.shape[0]))
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection='3d')
    
    if isinstance(estimate,type(None)):
        colors,min_,max_ = create_colormap(target,cmap="Greens")
        cmap_ = 'Greens'
        cbar_text = 'ABGM Tonnes'
    else:
        if not isinstance(target,type(np.array([]))):
            target = target.numpy()
        estimate = estimate.reshape(dims,dims)
        colors,min_,max_ = create_colormap(target,estimate)
        cmap_ = 'Reds'
        cbar_text = 'Absolute Error (Tonnes)'

    
    X = x.ravel()
    Y = y.ravel()
    Z = target.ravel()

    ax.bar3d(X, 
             Y, 
             np.zeros_like(target).ravel(), 
             1, 
             1, 
             Z, 
             shade=True,
             color=colors)

     
    ax.set_zlim(0, z_max)
    ax.set_zlabel('Above Ground Biomass (tonnes)')
    ax.set_title(title)

    sm = ScalarMappable(cmap=plt.get_cmap(cmap_), norm=plt.Normalize(vmin=min_, vmax=max_))
    sm.set_array([])

    # Create a colorbar
    cbar = plt.colorbar(sm, ax=ax, label=cbar_text)

    plt.savefig(savedir)
    plt.show()



def plot_performance_scatter(df,rmse='',savedir='./'):

    plt.style.use('fivethirtyeight')
    df = df.loc[(df['predicted'] < 400) & (df['ground_truth'] < 400)]
    df['error'] = np.abs(df['ground_truth'] - df['predicted'])

    xy = np.vstack([df['predicted'].values, df['ground_truth'].values])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots(figsize=(16, 9))
    scatter = ax.scatter(x=df['ground_truth'], y=df['error'], s=1, c=z)

    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Density', rotation=90)

    plt.xlabel('Ground Truth (tonnes)')
    plt.ylabel('Error (tonnes)')
    plt.title(f'Error-True Value w. KDE CMAP (RMSE: {rmse})')
    
    plt.savefig(savedir+'error_scatter.png')

    plt.show()



    df.ground_truth.hist(bins=50,figsize=(16,9), color='#00b894')
    plt.title(f'AGBM Distribution (RMSE: {rmse})')
    
    plt.savefig(savedir+'AGBM_hist.png')
    plt.show()