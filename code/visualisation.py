import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.signal import convolve2d
import sys
import parser


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


class Vizhelper():
    def init(self):
        pass

    def get_models_from_perfs(self):

        perfs = pd.read_csv('results/overall_performances.csv',sep=';')
        
        currently_compared = pd.read_csv('results/plots/currently_compared_models.csv')

        unq_models = perfs['model'].unique().tolist()

        self.models = {}
        self.performance = {}

        for model in unq_models:
            self.performance[model] = {}
            best_model = perfs.loc[perfs['model']==model].sort_values('RMSE_test').head(1)
            self.models[model] = best_model['runID'].values[0]

            self.performance[model]['RMSE_train']=best_model['RMSE_train'].values[0]
            self.performance[model]['RMSE_test']=best_model['RMSE_test'].values[0]

        identical = 0
        for model,runID in self.models.items():
            earlier_model_version = currently_compared.loc[currently_compared['model']==model]['runID']
            if earlier_model_version.shape[0]>0 and earlier_model_version.values[0]==runID:
                identical+=1

        if len(unq_models)==identical:
            print('No change has been made in the ranking of each modeltype')
            print('exiting without plotting')
            sys.exit()
            


    def aggregate_estimates(self,estimates):
        new_ests = {'predicted':[],
                'ground_truth':[]}
        
        for sample in range(50):
            for pixel in tqdm(range(256**2)):
                pixel = (sample*256**2*12)+pixel
                indexes = []
                mps = []
                for month in range(12):
                    monthly_pixel = pixel+256**2*month
                    indexes.append(monthly_pixel)
                    mps.append(monthly_pixel)

                unks = estimates.iloc[indexes]
                new_ests['predicted'].append(unks['predicted'].mean())
                new_ests['ground_truth'].append(unks['ground_truth'].mean())
        return pd.DataFrame(new_ests)
    

    def get_estimates(self):
        estimates = {}

        for model,runID in self.models.items():
            estimates[model] = pd.read_csv(f'results/{model}/{runID}/estimates.csv',index_col=0)
            if model == 'CNN':
                estimates[model] = self.aggregate_estimates(estimates[model])

        self.estimates = estimates

    def get_avg_diff_from_neighbours(self):

        self.model_dicts = {}

        stpsz = 256**2
        for model,estimate in self.estimates.items():
            plotting_dict = {'x':[],
                             'y':[]}
            
            for samp in range(50): 
                chip = estimate.iloc[samp*stpsz:(samp+1)*stpsz]
                print(chip.shape)


                # Create a random 256x256 numpy matrix as an example
                matrix = chip['ground_truth'].values.reshape(256,256)

                # Define a 3x3 kernel for averaging with the center element set to 0
                kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]]) / 8

                # Perform convolution to calculate the averages
                averaged_matrix = convolve2d(matrix, kernel, mode='same')

                neigh_diff = np.abs(matrix-averaged_matrix).flatten().tolist()

                abs_error = np.abs(chip['ground_truth'].values-chip['predicted'].values).tolist()
                plotting_dict['x']+=neigh_diff
                plotting_dict['y']+=abs_error

            
            self.model_dicts[model] = plotting_dict



    def get_max_diff_from_neighbours(self):
        pass

    def compare_spatial_performance(self):
        """
        
        """
        plt.figure(figsize=(16, 9))
        plt.style.use('fivethirtyeight')

        for model, coords in self.model_dicts.items():   
            # Add the regression lines for both scatterplots
            sns.regplot(x=coords['x'], y=coords['y'], scatter=False, label=model, line_kws={"linewidth": 1})


        # Add labels and a legend
        plt.xlabel('Mean target difference from neighbouring pixels (tonnes)')
        plt.ylabel('Absolute Error of model on given pixel (Tonnes)')
        plt.legend(loc='best')
            
            
        plt.xlim(0, 200)
        plt.ylim(0, 200)

        ax2 = plt.twinx()
        sns.kdeplot(coords['x'], color='blue', label='Pixel Density', shade=True)
        ax2.set_ylabel('Pixel Density')

        plt.title("Performance of Models on Pixels compared to Mean Neighbouring Pixel Difference")


        # Save the plot
        plt.savefig('results/plots/regression_kde.png')


    def compare_performance(self):
        
        categories = list(self.performance.keys())
        y_train = [self.performance[model]['RMSE_train'] for model in categories]
        y_test = [self.performance[model]['RMSE_test'] for model in categories]

        plt.figure(figsize=[16,9])
        plt.title('RMSE')
        plt.style.use('fivethirtyeight')
        

        # Set the width of each bar and the gap between groups
        bar_width = 0.35
        gap = 0.15

        # Calculate the x-positions for the bars
        x = np.arange(len(categories))

        # Create the grouped bar chart
        plt.bar(x - bar_width / 2, y_train, bar_width, label='Train', align='center')
        plt.bar(x + bar_width / 2, y_test, bar_width, label='Test', align='center')

        # Customize the chart
        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.title('Train and Test Performance for models')
        plt.xticks(x, categories)
        
        plt.legend()

        plt.savefig('results/plots/performance.png')
            
        
 

    def compare_emissions(self):
        for plot_type in ['training','inference']:
            x = []
            y = []
            y_comp = []
            for model,runID in self.models.items():
                kwh,g_co2_eq,comparison = parser.aggregate_consumption(f'results/{model}/{runID}/carbon_emissions/{plot_type}')
                x.append(model)
                y.append(g_co2_eq)
                y_comp.append(comparison['km travelled by car'])
                
            
            plt.figure(figsize=[16,9])
            plt.title(f'CO2 Eq Emissions during {plot_type}')
            plt.style.use('fivethirtyeight')
            plt.bar(x,y)
            plt.ylabel('CO2 Eq')
            
            plt.savefig(f'results/plots/co2_emissions_{plot_type}.png')
            
            
            
            plt.figure(figsize=[16,9])
            plt.title(f'Equivalent km in car during {plot_type}')
            plt.bar(x,y_comp)
            plt.ylabel(f'Km by car')
            plt.savefig(f'results/plots/energy_comparison_{plot_type}.png')


    def update_master_df(self):
        master_dict = {'model':[],
                       'runID':[]}

        for model,runID in self.models.items():
            master_dict['model'].append(model)
            master_dict['runID'].append(runID)

        
        pd.DataFrame(master_dict).to_csv('results/plots/currently_compared_models.csv')

def compare_best_models():
    # Hent Modeller fra Overall Performances
    Vizhelper_ = Vizhelper()

    Vizhelper_.get_models_from_perfs()

    Vizhelper_.get_estimates()

    # find forskellen for hver pixel fra gennemsnittet af deres naboer
    Vizhelper_.get_avg_diff_from_neighbours()

    # Lav regressionslinje for alle modeller
    Vizhelper_.compare_spatial_performance()

    # Performance bar plot
    Vizhelper_.compare_performance()

    # Emissons bar plot
    Vizhelper_.compare_emissions()

    Vizhelper_.update_master_df()
