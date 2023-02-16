from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer
from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
import sigfig
import pandas as pd
import numpy as np
from GravNN.Networks.Model import load_config_and_model
import matplotlib.pyplot as plt

class ExtrapolationMetrics(ExtrapolationVisualizer):
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)

    def calculate_error(self):
        train_end = self.experiment.losses['percent'][self.idx_test][(self.max_idx-20):self.max_idx]
        test_end = self.experiment.losses['percent'][self.idx_test][:-20]

        train_avg = sigfig.round(np.mean(train_end), sigfigs=2)
        train_std = sigfig.round(np.std(train_end), sigfigs=2)
        train_str = "%s ± %s" % (train_avg, train_std)
        
        test_avg = sigfig.round(np.mean(test_end), sigfigs=2)
        test_std = sigfig.round(np.mean(test_end), sigfigs=2)
        test_str = "%s ± %s" % (test_avg, test_std)

        #print("Train: ", train_str)
        #print("Test: ", test_str)

        k_before = self.experiment.config['tanh_k'][0]
        r_before = self.experiment.config['tanh_r'][0]
        k_after = self.experiment.model.trainable_variables[-2].numpy()[0]
        r_after = self.experiment.model.trainable_variables[-3].numpy()[0]

        #print("Tanh_k before: ", k_before)
        #print("Tanh_r before: ", r_before)
        #print("Tanh_k after: ", k_after)
        #print("Tanh_r after: ", r_after)

        return train_avg, test_avg, k_before, k_after, r_before, r_after

    def compare_error_graphs(self,df):
        k_bef = df["K_before"]
        r_bef = df["R_before"]
        k_af = df["K_after"]
        r_af = df["R_after"]
        test = df["Test"]

        plt.figure(1)
        plt.plot(k_bef, test, 'ro')
        plt.xlabel("K_before")
        plt.ylabel("Test")
        plt.title("K_before vs Test")
        plt.savefig("Kbef.png")

        plt.figure(2)
        plt.plot(r_bef, test, 'ro')
        plt.xlabel("R_before")
        plt.ylabel("Test")
        plt.title("R_before vs Test")
        plt.savefig("Rbef.png")

        plt.figure(3)
        plt.plot(k_af, test, 'ro')
        plt.xlabel("K_after")
        plt.ylabel("Test")
        plt.title("K_after vs Test")
        plt.savefig("Kaf.png")

        plt.figure(4)
        plt.plot(r_af, test, 'ro')
        plt.xlabel("R_after")
        plt.ylabel("Test")
        plt.title("R_after vs Test")
        plt.savefig("Raf.png")

        #plt.show()

def main():
    
    df = pd.read_pickle("Data/Dataframes/untrained2.data")
    metrics_df = pd.DataFrame(columns=['Train', 'Test', 'K_before', 'K_after', 'R_before', 'R_after'], index=range(1,len(df["id"])))
    file = "Data/Metrics_Test3.csv"
    counter = 0

    for id in df["id"].values:
        config, model = load_config_and_model(id, df)
        counter += 1

        # evaluate the error at "training" altitudes and beyond
        extrapolation_exp = ExtrapolationExperiment(model, config, 1000)
        extrapolation_exp.run()

        #vis = ExtrapolationVisualizer(extrapolation_exp, x_axis='dist_2_COM', plot_fcn=plt.semilogy)
        metrics = ExtrapolationMetrics(extrapolation_exp)

        train, test, k_bef, k_af, r_bef, r_af = metrics.calculate_error()

        metrics_df.at[counter, "Train"] = train
        metrics_df.at[counter, "Test"] = test
        metrics_df.at[counter, "K_before"] = k_bef
        metrics_df.at[counter, "K_after"] = k_af
        metrics_df.at[counter, "R_before"] = r_bef
        metrics_df.at[counter, "R_after"] = r_af

    metrics_df.to_csv(file, header=['Train', 'Test', 'K_before', 'K_after', 'R_before', 'R_after'])
    
    #metrics.compare_error_graphs(metrics_df)  


if __name__ == "__main__":
    main()

# Put all values into a dataframe
# Save dataframe to csv
# New function, load df, make some graphs for comparison: Test vs K_bef/af, Test vs R_bef/af, 3D graph???