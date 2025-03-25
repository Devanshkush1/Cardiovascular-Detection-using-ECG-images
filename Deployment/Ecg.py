from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from numpy import asarray
from skimage.metrics import structural_similarity
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
from natsort import natsorted
from sklearn import linear_model, tree, ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

class ECG:
    def __init__(self, uploaded_file, tmpdirname):
        """
        Initialize the ECG object with the uploaded file and a temporary directory.
        """
        self.uploaded_file = uploaded_file
        self.tmpdirname = tmpdirname

    def getImage(self):
        """
        This function gets user image
        return: user image
        """
        image = imread(self.uploaded_file)
        return image

    def GrayImgae(self, image):
        """
        This function converts the user image to Gray Scale
        return: Gray scale Image
        """
        image_gray = color.rgb2gray(image)
        image_gray = resize(image_gray, (1572, 2213))
        return image_gray

    def DividingLeads(self, image):
        """
        This Function Divides the ECG image into 13 Leads including long lead. Bipolar limb leads(Leads1,2,3). Augmented unipolar limb leads(aVR,aVF,aVL). Unipolar (+) chest leads(V1,V2,V3,V4,V5,V6)
        return: List containing all 13 leads divided
        """
        Lead_1 = image[300:600, 150:643]  # Lead 1
        Lead_2 = image[300:600, 646:1135]  # Lead aVR
        Lead_3 = image[300:600, 1140:1625]  # Lead V1
        Lead_4 = image[300:600, 1630:2125]  # Lead V4
        Lead_5 = image[600:900, 150:643]  # Lead 2
        Lead_6 = image[600:900, 646:1135]  # Lead aVL
        Lead_7 = image[600:900, 1140:1625]  # Lead V2
        Lead_8 = image[600:900, 1630:2125]  # Lead V5
        Lead_9 = image[900:1200, 150:643]  # Lead 3
        Lead_10 = image[900:1200, 646:1135]  # Lead aVF
        Lead_11 = image[900:1200, 1140:1625]  # Lead V3
        Lead_12 = image[900:1200, 1630:2125]  # Lead V6
        Lead_13 = image[1250:1480, 150:2125]  # Long Lead

        # All Leads in a list
        Leads = [Lead_1, Lead_2, Lead_3, Lead_4, Lead_5, Lead_6, Lead_7, Lead_8, Lead_9, Lead_10, Lead_11, Lead_12, Lead_13]
        fig, ax = plt.subplots(4, 3)
        fig.set_size_inches(10, 10)
        x_counter = 0
        y_counter = 0

        # Create 12 Lead plot using Matplotlib subplot
        for x, y in enumerate(Leads[:len(Leads)-1]):
            if (x+1) % 3 == 0:
                ax[x_counter][y_counter].imshow(y)
                ax[x_counter][y_counter].axis('off')
                ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
                x_counter += 1
                y_counter = 0
            else:
                ax[x_counter][y_counter].imshow(y)
                ax[x_counter][y_counter].axis('off')
                ax[x_counter][y_counter].set_title("Leads {}".format(x+1))
                y_counter += 1
        
        # Save the image to the temporary directory
        fig.savefig(os.path.join(self.tmpdirname, 'Leads_1-12_figure.png'))
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(10, 10)
        ax1.imshow(Lead_13)
        ax1.set_title("Leads 13")
        ax1.axis('off')
        fig1.savefig(os.path.join(self.tmpdirname, 'Long_Lead_13_figure.png'))

        return Leads

    def PreprocessingLeads(self, Leads):
        """
        This Function Performs preprocessing on the extracted leads.
        """
        fig2, ax2 = plt.subplots(4, 3)
        fig2.set_size_inches(10, 10)
        # Setting counter for plotting based on value
        x_counter = 0
        y_counter = 0

        for x, y in enumerate(Leads[:len(Leads)-1]):
            # Converting to gray scale
            grayscale = color.rgb2gray(y)
            # Smoothing image
            blurred_image = gaussian(grayscale, sigma=1)
            # Thresholding to distinguish foreground and background
            # Using otsu thresholding for getting threshold value
            global_thresh = threshold_otsu(blurred_image)

            # Creating binary image based on threshold
            binary_global = blurred_image < global_thresh
            # Resize image
            binary_global = resize(binary_global, (300, 450))
            if (x+1) % 3 == 0:
                ax2[x_counter][y_counter].imshow(binary_global, cmap="gray")
                ax2[x_counter][y_counter].axis('off')
                ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
                x_counter += 1
                y_counter = 0
            else:
                ax2[x_counter][y_counter].imshow(binary_global, cmap="gray")
                ax2[x_counter][y_counter].axis('off')
                ax2[x_counter][y_counter].set_title("pre-processed Leads {} image".format(x+1))
                y_counter += 1
        fig2.savefig(os.path.join(self.tmpdirname, 'Preprossed_Leads_1-12_figure.png'))

        # Plotting lead 13
        fig3, ax3 = plt.subplots()
        fig3.set_size_inches(10, 10)
        # Converting to gray scale
        grayscale = color.rgb2gray(Leads[-1])
        # Smoothing image
        blurred_image = gaussian(grayscale, sigma=1)
        # Thresholding to distinguish foreground and background
        # Using otsu thresholding for getting threshold value
        global_thresh = threshold_otsu(blurred_image)
        print(global_thresh)
        # Creating binary image based on threshold
        binary_global = blurred_image < global_thresh
        ax3.imshow(binary_global, cmap='gray')
        ax3.set_title("Leads 13")
        ax3.axis('off')
        fig3.savefig(os.path.join(self.tmpdirname, 'Preprossed_Leads_13_figure.png'))

    def SignalExtraction_Scaling(self, Leads):
        """
        This Function Performs Signal Extraction using various steps, techniques: convert to grayscale, apply gaussian filter, thresholding, perform contouring to extract signal image and then save the image as 1D signal
        """
        fig4, ax4 = plt.subplots(4, 3)
        # fig4.set_size_inches(10, 10)
        x_counter = 0
        y_counter = 0
        for x, y in enumerate(Leads[:len(Leads)-1]):
            # Converting to gray scale
            grayscale = color.rgb2gray(y)
            # Smoothing image
            blurred_image = gaussian(grayscale, sigma=0.7)
            # Thresholding to distinguish foreground and background
            # Using otsu thresholding for getting threshold value
            global_thresh = threshold_otsu(blurred_image)

            # Creating binary image based on threshold
            binary_global = blurred_image < global_thresh
            # Resize image
            binary_global = resize(binary_global, (300, 450))
            # Finding contours
            contours = measure.find_contours(binary_global, 0.8)
            contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
            for contour in contours:
                if contour.shape in contours_shape:
                    test = resize(contour, (255, 2))
            if (x+1) % 3 == 0:
                ax4[x_counter][y_counter].invert_yaxis()
                ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0], linewidth=1, color='black')
                ax4[x_counter][y_counter].axis('image')
                ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
                x_counter += 1
                y_counter = 0
            else:
                ax4[x_counter][y_counter].invert_yaxis()
                ax4[x_counter][y_counter].plot(test[:, 1], test[:, 0], linewidth=1, color='black')
                ax4[x_counter][y_counter].axis('image')
                ax4[x_counter][y_counter].set_title("Contour {} image".format(x+1))
                y_counter += 1
        
            # Scaling the data and testing
            lead_no = x
            scaler = MinMaxScaler()
            fit_transform_data = scaler.fit_transform(test)
            Normalized_Scaled = pd.DataFrame(fit_transform_data[:, 0], columns=['X'])
            Normalized_Scaled = Normalized_Scaled.T
            # Save scaled data to CSV in the temporary directory
            csv_path = os.path.join(self.tmpdirname, f'Scaled_1DLead_{lead_no+1}.csv')
            if os.path.isfile(csv_path):
                Normalized_Scaled.to_csv(csv_path, mode='a', index=False)
            else:
                Normalized_Scaled.to_csv(csv_path, index=False)
        
        fig4.savefig(os.path.join(self.tmpdirname, 'Contour_Leads_1-12_figure.png'))

    def CombineConvert1Dsignal(self):
        """
        This function combines all 1D signals of 12 Leads into one file csv for model input.
        returns the final dataframe
        """
        # First read the Lead1 1D signal
        test_final = pd.read_csv(os.path.join(self.tmpdirname, 'Scaled_1DLead_1.csv'))
        # Loop over all the 11 remaining leads and combine as one dataset using pandas concat
        for files in natsorted(os.listdir(self.tmpdirname)):
            if files.endswith(".csv"):
                if files != 'Scaled_1DLead_1.csv':
                    df = pd.read_csv(os.path.join(self.tmpdirname, files))
                    test_final = pd.concat([test_final, df], axis=1, ignore_index=True)

        return test_final
        
    def DimensionalReduciton(self, test_final):
        """
        This function reduces the dimensionality of the 1D signal using PCA
        returns the final dataframe
        """
        # First load the trained PCA model
        pca_loaded_model = joblib.load('model_pkl/PCA_ECG (1).pkl')
        result = pca_loaded_model.transform(test_final)
        final_df = pd.DataFrame(result)
        return final_df

    def ModelLoad_predict(self, final_df):
        """
        This Function Loads the pretrained model and performs ECG classification
        return the classification Type.
        """
        loaded_model = joblib.load('model_pkl/Heart_Disease_Prediction_using_ECG (4).pkl')
        result = loaded_model.predict(final_df)
        if result[0] == 1:
            return "You ECG corresponds to Myocardial Infarction"
        elif result[0] == 0:
            return "You ECG corresponds to Abnormal Heartbeat"
        elif result[0] == 2:
            return "Your ECG is Normal"
        else:
            return "You ECG corresponds to History of Myocardial Infarction"