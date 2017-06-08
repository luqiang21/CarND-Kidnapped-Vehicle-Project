/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;
	default_random_engine gen;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for(int i=0; i < num_particles; i++){
		Particle p;

		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;

		particles.push_back(p);
		weights.push_back(1);
	}


	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for(int i=0; i < num_particles; i++){
		// if yaw_rate is 0, the equation in the lesson cannot be used.
		if(yaw_rate == 0){
			particles[i].x += cos(particles[i].theta) * velocity + dist_x(gen);
			particles[i].y += sin(particles[i].theta) * velocity + dist_y(gen);
			particles[i].theta += dist_theta(gen);

		}else{
			double new_theta = particles[i].theta + yaw_rate * delta_t + dist_theta(gen);
			double coeff = velocity / yaw_rate;

			particles[i].x += coeff * (sin(new_theta) - sin(particles[i].theta)) + dist_x(gen);
			particles[i].y += coeff * (cos(particles[i].theta) - cos(new_theta)) + dist_y(gen);
			particles[i].theta = new_theta;

		}
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	double distance;

	for(int i=0; i < observations.size(); i++){
		double min_dist = std::numeric_limits<double>::max();

		for(int j=0; j < predicted.size(); j++){
			distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if(distance < min_dist){
				min_dist = distance;
				observations[i].id = predicted[j].id;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double std_x_2 = pow(std_landmark[0], 2);
	double std_y_2 = pow(std_landmark[1], 2);
	double std_x_y = std_landmark[0] * std_landmark[1];

	for(int i=0; i < num_particles; i++){
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// predict landmarks within sensor_range
		vector <LandmarkObs> predicted;

		for (int j=0; j < map_landmarks.landmark_list.size(); j++){

      int id = map_landmarks.landmark_list[j].id_i;
      double land_x = map_landmarks.landmark_list[j].x_f;
      double land_y = map_landmarks.landmark_list[j].y_f;

      if (dist(p_x, p_y, land_x, land_y) < sensor_range) {
				LandmarkObs landmark;
				landmark.id = id;
				landmark.x = land_x;
				landmark.y = land_y;

        predicted.push_back(landmark);
      }

		// transformation based on p_x, p_y, p_theta and observed landmark positions.
		vector <LandmarkObs> observations_transformed;
		for (int k=0; k < observations.size(); k++){
			LandmarkObs observation_transformed;
			observation_transformed.id = observations[i].id;
			observation_transformed.x = observations[i].x * cos(p_theta)
																- observations[i].y * sin(p_theta) + p_x;
			observation_transformed.y = observations[i].x * sin(p_theta)
																- observations[i].y * cos(p_theta) + p_y;

			observations_transformed.push_back(observation_transformed);
		}

		// Association
    dataAssociation(predicted, observations_transformed);

		// Update weight
		double final_weight = 1.0;
		for (int l=0; l < observations.size(); l++){
			int ID = observations_transformed[l].id;
			double diff_x = observations_transformed[l].x -	predicted[ID].x;
			double diff_x_2 = diff_x * diff_x;
			double diff_y = observations_transformed[l].y -	predicted[ID].y;
			double diff_y_2 = diff_y * diff_y;

			double exponent = (diff_x_2 / (2*std_x_2) + diff_y_2 / (2*std_y_2));
			final_weight *= 1/ (2*M_PI*std_x_y) * exp(exponent);
		}

		particles[i].weight = final_weight;
		weights[i] = final_weight;

	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// http://en.cppreference.com/w/cpp/algorithm/max_element use max_element to
	// obtain maximum element in the vector
	default_random_engine gen;

	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_real_distribution <double> dist_beta(0.0, 2 * max_weight);
	double beta = 0.0;
	int index = rand() % num_particles;

	std::vector <Particle> particles_resampled;

	// wheel algorithm
	for(int i=0; i < num_particles; i++){
		beta += dist_beta(gen);

		while(beta > weights[index]){
			beta -= weights[index];
			index = (index + 1) % num_particles;

		}
		particles_resampled.push_back(particles[index]);
	}

	particles = particles_resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
