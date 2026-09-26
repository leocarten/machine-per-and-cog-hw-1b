import random
import numpy as np

class Homework:
    def __init__(self):
        self.dataSet = {
            "w1": {
                "x1": [-5.01, -5.43, 1.08, 0.86, -2.67, 4.94, -2.51, -2.25, 5.56, 1.03],
                "x2": [-8.12, -3.48, -5.52, -3.78, 0.63, 3.29, 2.09, -2.13, 2.86, -3.33],
                "x3": [-3.68, -3.54, 1.66, -4.11, 7.39, 2.08, -2.59, -6.94, -2.26, 4.33]
            },

            "w2": {
                "x1": [-0.91, 1.30, -7.75, -5.47, 6.14, 3.60, 5.37, 7.18, -7.39, -7.50],
                "x2": [-0.18, -2.06, -4.54, 0.50, 5.72, 1.26, -4.63, 1.46, 1.17, -6.32],
                "x3": [-0.05, -3.53, -0.95, 3.92, -4.85, 4.36, -3.65, -6.66, 6.30, -0.31]
            },

            "w3": {
                "x1": [5.35, 5.12, -1.34, 4.48, 7.11, 7.17, 5.75, 0.77, 0.90, 3.52],
                "x2": [2.26, 3.22, -5.31, 3.42, 2.39, 4.33, 3.97, 0.27, -0.43, -0.36],
                "x3": [8.13, -2.66, -9.87, 5.19, 9.21, -0.98, 6.65, 2.41, -8.71, 6.43]
            }
        }
        
    def programOutputDelimiter(self):
         return "------------------------------------------------------------------------------------------"

    def generateNormalDistribution(self, mean: float, standardDeviation: float, dimensions: int) -> [int]:
        normalDistribution = []
        for i in range(dimensions):
             number = random.gauss(mean, standardDeviation)
             normalDistribution.append(number)
        return normalDistribution
    
    def question_1a(self):
        # Things to know: 
            # d dimensions just means how many inputs you have (in our case, 3)
            # N(µ,Σ) -> this tells us 2 things about the distributions, µ is the mean and Σ is the covariance matrix (how spread out the data is)
            # lets use an example to help understand µ and Σ: let's say we are trying to classify fish. x1 is length (in inches) and x2 is weight (in pounds). 
                # lets say µ is [10, 2]: this means a fish is on average 10inches long and 2lbs in weight
                # lets say Σ is [ 2, 0.5, 0.5, 0.25 ]: the 2 represents the variance in length, the 0.25 is the varience in weight. 
                    # (2 is calculated by the average distance from the mean squared, so if 5 lengths were 8,9,10,11,12 -> -2,-1,0,1,2 but then squared -> 4,1,0,1,4 -> added together 10 -> mean is 2)
                    # the 0.25 is the same but for weight (average squared mean)
                    # now, what the 0.5 tells us is how the variables are related. since the number is positive, it tells us that as the lenght increases, weight also increases. same vice versa, and weight increase, length tends to increase.
                    # so [ 2, 0.5, 0.5, 0.25 ] pretty much = [variance of length, covar of length and weight, covar of weight and length, variance of weight

            print("Question 1a: Write a procedure to generate random samples according to a normal distribution N(µ,Σ) in d dimensions.")
            # Algorithm steps
            # Step 1: Make a basic random sample based on the mean (0) and SD (1) in d dimensions (in our case, 3 dimensions since there are 3 inputs). We will call this 'z'.
            z = self.generateNormalDistribution(0, 1, 2)
            z = np.array(z)
            print(f"z={z}")

            # Step 2: Reshape datapoints in 'z' to have a relationship similiar to Σ
            # for simplicity sake, i am going to assign the covariance_matrix as: [ 2, 0.5, 0.5, 0.25 ]
            covariance_matrix = np.array([
                                            [2, 0.5],
                                            [0.5, 0.25]
                                        ]) # 2=variance of fish length, 0.5=relationship between length and weight, 0.25=varience of fish weight
            print(f"covar_matrix={covariance_matrix}")
            # now, find A such that AA' = Σ
            # it's important to knpw that A is the "tool" we use to reshape the data points in z to match the covariance in Σ
            A = np.linalg.cholesky(covariance_matrix)
            print(f"A={A}")

            # Step 3: calculate Az. Az is just taking the random sample (z) and re-shaping it to match the specified covariance of the variables
            Az = A @ z
            print(f"Az={Az}")

    def g(self, _class, features, part_of_equation):
         score = part_of_equation[0]@part_of_equation[1] - part_of_equation[2] - part_of_equation[3] + part_of_equation[4]
         print(f"Score for class '{_class}' based on inputs length={features[0]} and weight={features[1]}: {score}")
         return score
    
    def question_1b(self):
        print("Question 1b: Write a procedure to calculate the discriminant function (of the form given in Eq. 47) for a given normal distribution and prior probability P(wi).")
        # Eq. 47: gi(x) =−1/2(x− µi)t * Σ−1i(x− µi)−d/2 ln 2π −1/2ln |Σi| +ln P(ωi)
        # Things to know:
            # discriminant function: a DF is assigns a score to tell you how well the data point 'x' fits into a class
                # for example, if g_salmon(x) = 5 and g_bass(x) = 2, we would classify x as a salmon since it has a higher score
            # normal distribution: a ND is a set of data that is centered around the mean and has data progressively getting stretched to each side of the mean (positive and negative standard deviations)
            # prior probability: prior probability is the probability that x belongs to a class before looking at the data.
                # for example, lets go back to the salmon / bass probelm. lets say that based on the general population, 60% of our fish are salmon while 40% bass. 

        # Let's say we are trying to classify salmon and bass. This is what the question is asking: write a function that calculates a score for how well a data point 'x' fits as a salmon or bass.

        print("For this HW question I will be using 2 classes: Salmon and Bass. I will be using 2 inputs: Length and weight. I will be using 'made-up' length and weights for each fish for training data.")

        # Lets now break down the Eq. 47:
        # −1/2(x− µi)t
            # x: the feature input
            # µi: the mean of the class
            # transpose that maxtrix
            # then multiply by -1/2
        salmon_length = [24, 26, 27, 28, 29, 30, 32]           # inches
        salmon_weight = [5.5, 7.0, 8.0, 9.0, 10.0, 11.5, 14.0] # pounds
        salmon_length_mean = sum(salmon_length) / len(salmon_length)
        salmon_weight_mean = sum(salmon_weight) / len(salmon_weight)
        print(f"Observed salmon length: {salmon_length}")
        print(f"Observed salmon weight: {salmon_weight}")
        print(f"Salmon length mean: {salmon_length_mean}")
        print(f"Salmon weight mean: {salmon_weight_mean}")

        bass_length = [12, 13, 14, 15, 16, 17, 18]        # inches
        bass_weight = [0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.7] # pounds
        bass_length_mean = sum(bass_length) / len(bass_length)
        bass_weight_mean = sum(bass_weight) / len(bass_weight)
        print(f"Observed bass length: {bass_length}")
        print(f"Observed bass weight: {bass_weight}")
        print(f"bass length mean: {bass_length_mean}")
        print(f"bass weight mean: {bass_weight_mean}")

        # NOW, we are seeing a new fish with length 22 inches, weight 4.9 lbs
        x = np.array([22, 4.9]) 

        salmon_mean = np.array([
            salmon_length_mean,
            salmon_weight_mean
        ])

        bass_mean = np.array([
            bass_length_mean,
            bass_weight_mean
        ])

        # perform our subtraction
        x_minus_salmon_mean = x - salmon_mean
        x_minus_bass_mean = x - bass_mean

        # transpose each matrix
        x_minus_salmon_mean_transposed = -0.5*(x_minus_salmon_mean.T)
        x_minus_bass_mean_transposed = -0.5*(x_minus_bass_mean.T)
        part1_salmon = x_minus_salmon_mean_transposed
        part1_bass = x_minus_bass_mean_transposed

        # Σ−1i(x− µi)
            # Σ−1i: this is the inverse of the covariance matrix
            # x: the feature input
            # µi: the mean of the class
        salmon_data = np.array([
                                salmon_length,
                                salmon_weight
                            ])
        
        bass_data = np.array([
                                bass_length,
                                bass_weight
                            ])
        sigma_salmon = np.cov(salmon_data, bias=True)
        sigma_salmon_inv = np.linalg.inv(sigma_salmon)

        sigma_bass = np.cov(bass_data, bias=True)
        sigma_bass_inv = np.linalg.inv(sigma_bass)

        part2_bass = sigma_bass_inv @ x_minus_bass_mean
        part2_salmon = sigma_salmon_inv @ x_minus_salmon_mean

        # −d/2 ln 2π (the - is taken care of in g() function)
        d = len(salmon_data)
        part3 = (1*(d/2) * np.log( 2 * np.pi ))

        # −1/2ln |Σi| -> (the - is taken care of in g() function)
            # NOTE: the vertical bars are asking us to find the determinant of the covariant matrix
        part4_bass = 0.5 * np.log( np.linalg.det(sigma_bass) )
        part4_salmon = 0.5 * np.log( np.linalg.det(sigma_salmon) )

        # +ln P(ωi)
            # rememebr, prior probility is information about the genrral population NOT about our data. e.g., based on the general population, 60% of our fish are salmon while 40% bass. 
        prior_probability_of_salmon = 0.6
        prior_probability_of_bass = 0.4
        part5_salmon = np.log(prior_probability_of_salmon)
        part5_bass = np.log(prior_probability_of_bass)
        print(f"Prior probability for salmon: {prior_probability_of_salmon * 100}%")
        print(f"Prior probability for bass: {prior_probability_of_bass * 100}%")

        # calculate the scores and print
        salmon_score = self.g("Salmon", x, [part1_salmon, part2_salmon, part3, part4_salmon, part5_salmon])
        bass_score = self.g("Bass", x, [part1_bass, part2_bass, part3, part4_bass, part5_bass])

        # print results
        if salmon_score > bass_score:
            print(f"Based on scores, we will assign inputs length={x[0]} and weight={x[1]} to Salmon class.")
        else:
            print(f"Based on scores, we will assign inputs length={x[0]} and weight={x[1]} to Bass class.")

    def question_1c(self):
        print("Question 1c: Write a procedure to calculate the Euclidean distance between two arbitrary points.")
        # things to know: 
            # Euclidean distance is just a fancy way of saying the "straight line" distance between 2 points.
                # e.g., if you have two points (1,2) and (3,4) -> plot these 2 points in 2 dimensional space and then draw a line to connect these 2 points. then take the length of that line
            # formula for 2D Euclidean distance: sqrt[ (x2-x1)^2 + (y2-y1)^2 ]
            # the Euclidean distance between points may be relevant to tell someone how closely related "things" are. e.g., in KNN you might use a euclidean distance to see how far away a point is from "clusters" to then catehgorize your input.
        
        # implementation:
        p = [2,6]
        q = [-4,10]
        print(f"let p=({p[0]},{p[1]})")
        print(f"let q=({q[0]},{q[1]})")
        euclidean_distance = np.sqrt( (p[0] - q[0])**2 + (p[1] - q[1])**2 )
        print(f"The euclidean distance between p and q is: {euclidean_distance}")

    def question_1d(self):
        print("Question 1d: Write a procedure to calculate the Mahalanobis distance between the mean µ and an arbitrary point x, given the covariance matrix Σ.")
        # Things to know:
            # the Mahalanobis distance tells us how far away a data point is from the clustering of data, which takes into account the mean and variance of data
                # it says: How far away is it relative to what is normal for this group
            # this is an important distinction from euclidean distance bc mahalabonis takes into account vairance
            # basically, this can help us identify anomolies in data patterns
        
        # formula: sqrt[ (x-µ)T Σ-1 (x-µ) ] -> basically (inputs - mean)' * inverse_of_covariance * (inputs - mean)
            # x -> the point you want to test (the input)
            # μ -> the center / average of the group
        
        salmon_length = [24, 26, 27, 28, 29, 30, 32]           # inches
        salmon_weight = [5.5, 7.0, 8.0, 9.0, 10.0, 11.5, 14.0] # pounds
        salmon_data = np.array([
            salmon_length,
            salmon_weight
        ])
        salmon_length_mean = sum(salmon_length) / len(salmon_length)
        salmon_weight_mean = sum(salmon_weight) / len(salmon_weight)
        salmon_mean = np.array([
            salmon_length_mean,
            salmon_weight_mean
        ])
        sigma_salmon = np.cov(salmon_data, bias=True)
        sigma_salmon_inv = np.linalg.inv(sigma_salmon)

        # NOW, we are seeing a new fish with length 24 inches, weight 10 lbs
        x = np.array([24, 10]) 

        mahalanobis_distance = np.sqrt( (x-salmon_mean).T @ sigma_salmon_inv @ (x-salmon_mean) )

        print(f"Observed salmon length: {salmon_length}")
        print(f"Observed salmon weight: {salmon_weight}")
        print(f"Salmon length mean: {salmon_length_mean}")
        print(f"Salmon weight mean: {salmon_weight_mean}")
        print(f"New fish (x) beeing seen: length={x[0]}, weight={x[1]}")
        print(f"The new fish mahalanobis distance is: {mahalanobis_distance}")
    
def main():
    hw = Homework()

    print(hw.programOutputDelimiter())
    hw.question_1a()

    print(hw.programOutputDelimiter())
    hw.question_1b()

    print(hw.programOutputDelimiter())
    hw.question_1c()

    print(hw.programOutputDelimiter())
    hw.question_1d()
    print(hw.programOutputDelimiter())

main()