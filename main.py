import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import emcee
import corner
import qiskit
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Pauli
from qiskit.utils import QuantumInstance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import chi2

def generate_exotic_hadron_data(time, hadron_type='tetraquark'):
    if hadron_type == 'tetraquark':
        correlation = np.exp(-time / 5) + 0.5 * np.exp(-time / 10) * np.cos(time / 2)
    elif hadron_type == 'pentaquark':
        correlation = np.exp(-time / 7) + 0.3 * np.exp(-time / 15) * np.sin(time / 3)
    correlation += np.random.normal(scale=0.05, size=len(time))
    return correlation

def generate_coupled_channel_data(time, channels=2):
    base_functions = [
        lambda t, a, b: a * np.exp(-b * t),
        lambda t, a, b, c: a * np.exp(-b * t) * np.cos(c * t),
        lambda t, a, b, c: a * np.exp(-b * t) * np.sin(c * t)
    ]
    data = np.zeros((len(time), channels))
    for i in range(channels):
        func = np.random.choice(base_functions)
        params = np.random.rand(func.__code__.co_argcount - 1)
        data[:, i] = func(time, *params)
    coupling_matrix = np.random.rand(channels, channels)
    coupled_data = np.dot(data, coupling_matrix)
    coupled_data += np.random.normal(0, 0.05, coupled_data.shape)
    return coupled_data

time = np.arange(0, 100, 1)
tetraquark_data = generate_exotic_hadron_data(time, 'tetraquark')
pentaquark_data = generate_exotic_hadron_data(time, 'pentaquark')
coupled_data = generate_coupled_channel_data(time)

def eft_model_tetraquark(t, a, b, c, d, e):
    return a * np.exp(-b * t) + c * np.exp(-d * t) * np.cos(e * t)

def eft_model_pentaquark(t, a, b, c, d, e):
    return a * np.exp(-b * t) + c * np.exp(-d * t) * np.sin(e * t)

def coupled_eft_model(t, *params):
    channels = len(params) // 5
    result = np.zeros((len(t) * channels,))
    for i in range(channels):
        a, b, c, d, e = params[i*5:(i+1)*5]
        result[i::channels] = a * np.exp(-b * t) + c * np.exp(-d * t) * np.cos(e * t)
    return result

# Generate coupled data
coupled_data = generate_coupled_channel_data(time)

# Prepare initial guess based on the data
channels = coupled_data.shape[1]
initial_guess = []
for i in range(channels):
    channel_data = coupled_data[:, i]
    max_val = np.max(channel_data)
    min_val = np.min(channel_data)
    initial_guess.extend([max_val, 0.1, (max_val - min_val)/2, 0.05, 0.5])

# Increase maxfev and use better bounds
from scipy.optimize import curve_fit

params_coupled, _ = curve_fit(
    coupled_eft_model, 
    time, 
    coupled_data.flatten(), 
    p0=initial_guess,
    maxfev=10000,  # Increase maximum function evaluations
    bounds=([0]*len(initial_guess), [np.inf]*len(initial_guess))  # All parameters non-negative
)

# Print the fitted parameters
print("Fitted parameters for coupled model:")
for i in range(channels):
    print(f"Channel {i+1}: {params_coupled[i*5:(i+1)*5]}")

# Plot the results
plt.figure(figsize=(12, 6))
for i in range(channels):
    plt.plot(time, coupled_data[:, i], '.', label=f'Data Channel {i+1}')
    fitted_data = coupled_eft_model(time, *params_coupled)[i::channels]
    plt.plot(time, fitted_data, '-', label=f'Fit Channel {i+1}')
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.title('Coupled-Channel EFT Fit')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(time, tetraquark_data, 'b-', label='Tetraquark Data')
plt.plot(time, eft_model_tetraquark(time, *params_tetraquark), 'r--', label='Tetraquark Fit')
plt.plot(time, pentaquark_data, 'g-', label='Pentaquark Data')
plt.plot(time, eft_model_pentaquark(time, *params_pentaquark), 'm--', label='Pentaquark Fit')
plt.xlabel('Time')
plt.ylabel('Correlation Function')
plt.title('Exotic Hadron Data and EFT Fits')
plt.legend()
plt.grid(True)
plt.show()

def log_likelihood_exotic(params, t, y, yerr, model):
    return -0.5 * np.sum(((y - model(t, *params)) / yerr) ** 2)

def log_prior_exotic(params):
    a, b, c, d, e = params
    if 0 < a < 10 and 0 < b < 1 and 0 < c < 1 and 0 < d < 1 and 0 < e < np.pi:
        return 0.0
    return -np.inf

def log_posterior_exotic(params, t, y, yerr, model):
    return log_prior_exotic(params) + log_likelihood_exotic(params, t, y, yerr, model)

nwalkers, ndim = 50, 5
initial_params_tetraquark = params_tetraquark
p0_tetraquark = [initial_params_tetraquark + 1e-4 * np.random.randn(ndim) for _ in range(nwalkers)]
sampler_tetraquark = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_exotic, 
                                           args=(time, tetraquark_data, 0.05 * np.ones_like(tetraquark_data), eft_model_tetraquark))

sampler_tetraquark.run_mcmc(p0_tetraquark, 2000, progress=True)

samples_tetraquark = sampler_tetraquark.get_chain(discard=200, thin=15, flat=True)
corner.corner(samples_tetraquark, labels=["a", "b", "c", "d", "e"], truths=params_tetraquark)
plt.suptitle("Tetraquark Parameter Posterior Distributions")
plt.show()

def advanced_quark_circuit(params, num_qubits=6):
    qc = EfficientSU2(num_qubits, reps=3, entanglement='full')
    bound_circuit = qc.bind_parameters(params)
    return bound_circuit

def cost_function(params):
    qc = advanced_quark_circuit(params)
    expectation = sum([(-1)**bin(i).count('1') * np.abs(complex(qc.bind_parameters(params).evolve(basis_state).data[0]))**2
for i, basis_state in enumerate(qc.bind_parameters(params).qregs[0].states)])
    return expectation.real

num_qubits = 6

ansatz = EfficientSU2(num_qubits, reps=3, entanglement='full')

quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))

vqe = VQE(ansatz=ansatz, optimizer=SPSA(maxiter=200), quantum_instance=quantum_instance)

hamiltonian = sum([PauliSumOp(Pauli('XXIIII')), PauliSumOp(Pauli('YYIIII')), PauliSumOp(Pauli('ZZIIII')),
                   PauliSumOp(Pauli('IIXXII')), PauliSumOp(Pauli('IIYYII')), PauliSumOp(Pauli('IIZZII')),
                   PauliSumOp(Pauli('IIIIZZ')), PauliSumOp(Pauli('IIIIYY')), PauliSumOp(Pauli('IIIIZZ'))])

initial_point = np.random.rand(ansatz.num_parameters) * 2 * np.pi
vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian, initial_point=initial_point)

optimized_params = vqe_result.optimal_point

print(f"VQE Energy: {vqe_result.eigenvalue.real}")

X = np.vstack([time, tetraquark_data, pentaquark_data]).T
y = np.column_stack([tetraquark_data, pentaquark_data])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', max_iter=1000)
mlp_model.fit(X_train, y_train)

y_pred = mlp_model.predict(X_test)

plt.figure(figsize=(12, 6))
plt.scatter(y_test[:, 0], y_pred[:, 0], c='blue', label='Tetraquark')
plt.scatter(y_test[:, 1], y_pred[:, 1], c='red', label='Pentaquark')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Neural Network Predictions for Exotic Hadrons')
plt.legend()
plt.grid(True)
plt.show()

feature_importance = np.abs(mlp_model.coefs_[0])
feature_importance = feature_importance.sum(axis=1)
feature_importance /= feature_importance.sum()

plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel('PCA Components')
plt.ylabel('Importance')
plt.title('Feature Importance in Exotic Hadron Prediction')
plt.show()

print("Tetraquark parameters:", params_tetraquark)
print("Pentaquark parameters:", params_pentaquark)
print("Most important PCA component for prediction:", np.argmax(feature_importance))

class EFTRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.params = None

    def fit(self, X, y):
        def model(t, a, b, c, d, e):
            return a * np.exp(-b * t) + c * np.exp(-d * t) * np.cos(e * t)
        
        self.params, _ = curve_fit(model, X[:, 0], y)
        return self

    def predict(self, X):
        def model(t, a, b, c, d, e):
            return a * np.exp(-b * t) + c * np.exp(-d * t) * np.cos(e * t)
        
        return model(X[:, 0], *self.params)

def chi_square_test(observed, expected, errors):
    chi_sq = np.sum(((observed - expected) / errors) ** 2)
    dof = len(observed) - len(params_tetraquark)
    p_value = 1 - chi2.cdf(chi_sq, dof)
    return chi_sq, p_value

def aic_bic(observed, predicted, num_params):
    n = len(observed)
    residuals = observed - predicted
    sse = np.sum(residuals**2)
    aic = 2 * num_params + n * np.log(sse/n)
    bic = np.log(n) * num_params + n * np.log(sse/n)
    return aic, bic

chi_sq_tetraquark, p_value_tetraquark = chi_square_test(
    tetraquark_data, 
    eft_model_tetraquark(time, *params_tetraquark), 
    0.05 * np.ones_like(tetraquark_data)
)
print(f"Tetraquark Chi-Square: {chi_sq_tetraquark}, p-value: {p_value_tetraquark}")

rmse_tetraquark = np.sqrt(mean_squared_error(tetraquark_data, eft_model_tetraquark(time, *params_tetraquark)))
print(f"Tetraquark RMSE: {rmse_tetraquark}")

r2_tetraquark = r2_score(tetraquark_data, eft_model_tetraquark(time, *params_tetraquark))
print(f"Tetraquark R-squared: {r2_tetraquark}")

aic_tetraquark, bic_tetraquark = aic_bic(
    tetraquark_data, 
    eft_model_tetraquark(time, *params_tetraquark), 
    len(params_tetraquark)
)
print(f"Tetraquark AIC: {aic_tetraquark}, BIC: {bic_tetraquark}")

cv_scores = cross_val_score(
    EFTRegressor(),
    X=np.column_stack([time, tetraquark_data]),
    y=tetraquark_data,
    cv=5,
    scoring='neg_mean_squared_error'
)

print(f"Cross-validation MSE scores: {-cv_scores}")
print(f"Mean CV MSE: {-cv_scores.mean()}, Std: {cv_scores.std()}")

def predict_with_uncertainty(samples, time, model):
    predictions = np.array([model(time, *params) for params in samples])
    mean_prediction = np.mean(predictions, axis=0)
    std_prediction = np.std(predictions, axis=0)
    return mean_prediction, std_prediction

mean_tetraquark, std_tetraquark = predict_with_uncertainty(samples_tetraquark, time, eft_model_tetraquark)

plt.figure(figsize=(12, 6))
plt.plot(time, tetraquark_data, 'b.', label='Data')
plt.plot(time, mean_tetraquark, 'r-', label='Mean Prediction')
plt.fill_between(time, mean_tetraquark - 2*std_tetraquark, mean_tetraquark + 2*std_tetraquark, color='r', alpha=0.3, label='95% CI')
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.title('Tetraquark Prediction with Uncertainty')
plt.legend()
plt.show()

def compare_models(models, data, time):
    results = []
    for name, model in models.items():
        if isinstance(model, type) and issubclass(model, BaseEstimator):
            estimator = model()
            estimator.fit(np.column_stack([time]), data)
            predictions = estimator.predict(np.column_stack([time]))
            num_params = len(estimator.params)
        else:
            params, _ = curve_fit(model, time, data)
            predictions = model(time, *params)
            num_params = len(params)
        
        aic, bic = aic_bic(data, predictions, num_params)
        results.append((name, aic, bic))
    return results

models = {
    'EFT Tetraquark': EFTRegressor,
    'Simple Exponential': lambda t, a, b: a * np.exp(-b * t),
    'Double Exponential': lambda t, a, b, c, d: a * np.exp(-b * t) + c * np.exp(-d * t)
}

model_comparison = compare_models(models, tetraquark_data, time)
for name, aic, bic in model_comparison:
    print(f"{name}: AIC = {aic}, BIC = {bic}")

X_coupled = np.column_stack([time] + [coupled_data[:, i] for i in range(coupled_data.shape[1])])
y_coupled = coupled_data

X_coupled_scaled = scaler.fit_transform(X_coupled)

X_coupled_pca = pca.fit_transform(X_coupled_scaled)

X_train_coupled, X_test_coupled, y_train_coupled, y_test_coupled = train_test_split(X_coupled_pca, y_coupled, test_size=0.2, random_state=42)
mlp_model_coupled = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', max_iter=1000)
mlp_model_coupled.fit(X_train_coupled, y_train_coupled)

y_pred_coupled = mlp_model_coupled.predict(X_test_coupled)

plt.figure(figsize=(12, 6))
for i in range(y_test_coupled.shape[1]):
    plt.scatter(y_test_coupled[:, i], y_pred_coupled[:, i], label=f'Channel {i+1}')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Neural Network Predictions for Coupled-Channel System')
plt.legend()
plt.show()

def coupled_eft_model_advanced(t, *params):
    channels = len(params) // 7
    result = np.zeros((len(t), channels))
    for i in range(channels):
        a, b, c, d, e, f, g = params[i*7:(i+1)*7]
        result[:, i] = a * np.exp(-b * t) + c * np.exp(-d * t) * np.cos(e * t) + f * np.sin(g * t)
    return result.flatten()

initial_guess_advanced = np.random.rand(14)
params_coupled_advanced, _ = curve_fit(coupled_eft_model_advanced, np.tile(time, 2), coupled_data.flatten(), p0=initial_guess_advanced)

def log_probability_advanced(params):
    if np.any(params < 0):
        return -np.inf
    model = coupled_eft_model_advanced(time, *params)
    return -0.5 * np.sum((coupled_data.flatten() - model)**2 / (0.05**2))

nwalkers_advanced, ndim_advanced = 50, len(params_coupled_advanced)
p0_advanced = [params_coupled_advanced + 1e-4*np.random.randn(ndim_advanced) for _ in range(nwalkers_advanced)]

sampler_advanced = emcee.EnsembleSampler(nwalkers_advanced, ndim_advanced, log_probability_advanced)
state_advanced = sampler_advanced.run_mcmc(p0_advanced, 2000)

samples_advanced = sampler_advanced.get_chain(flat=True, discard=500, thin=15)
mean_prediction_advanced, std_prediction_advanced = predict_with_uncertainty(samples_advanced, time, coupled_eft_model_advanced)

plt.figure(figsize=(12, 6))
for i in range(coupled_data.shape[1]):
    plt.plot(time, coupled_data[:, i], '.', label=f'Data Channel {i+1}')
    plt.plot(time, mean_prediction_advanced[i::2], '-', label=f'Advanced Mean Prediction Channel {i+1}')
    plt.fill_between(time, mean_prediction_advanced[i::2] - 2*std_prediction_advanced[i::2], 
                     mean_prediction_advanced[i::2] + 2*std_prediction_advanced[i::2], alpha=0.3)
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.title('Advanced Coupled-Channel EFT Fit with Uncertainty')
plt.legend()
plt.show()

print("Coupled-channel advanced parameters:", params_coupled_advanced)
print("VQE optimized parameters:", optimized_params)

chi_sq_coupled, p_value_coupled = chi_square_test(
    coupled_data.flatten(), 
    coupled_eft_model_advanced(time, *params_coupled_advanced), 
    0.05 * np.ones_like(coupled_data.flatten())
)
print(f"Coupled-channel Chi-Square: {chi_sq_coupled}, p-value: {p_value_coupled}")

rmse_coupled = np.sqrt(mean_squared_error(coupled_data.flatten(), coupled_eft_model_advanced(time, *params_coupled_advanced)))
print(f"Coupled-channel RMSE: {rmse_coupled}")

r2_coupled = r2_score(coupled_data.flatten(), coupled_eft_model_advanced(time, *params_coupled_advanced))
print(f"Coupled-channel R-squared: {r2_coupled}")

aic_coupled, bic_coupled = aic_bic(
    coupled_data.flatten(), 
    coupled_eft_model_advanced(time, *params_coupled_advanced), 
    len(params_coupled_advanced)
)
print(f"Coupled-channel AIC: {aic_coupled}, BIC: {bic_coupled}")

print("Analysis complete.")