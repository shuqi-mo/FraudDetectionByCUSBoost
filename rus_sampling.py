import numpy as np

def rus_sampler(X_train, y_train):
    selected_idx = []
    selected_idx = np.asarray(selected_idx)

    value, counts = np.unique(y_train, return_counts=True)
    minority_class = value[np.argmin(counts)]
    majority_class = value[np.argmax(counts)]

    idx_min = np.where(y_train == minority_class)[0]
    idx_maj = np.where(y_train == majority_class)[0]

    majority_class_instances = X_train[idx_maj]
    majority_class_labels = y_train[idx_maj]

    minority_class_instances = X_train[idx_min]
    minority_class_labels = y_train[idx_min]

    selected_idx = np.hstack((np.random.choice(np.asarray(idx_maj), size=len(minority_class_instances)), idx_min))
    np.random.shuffle(selected_idx)

    X_sampled = X_train[selected_idx]
    y_sampled = y_train[selected_idx]
    return X_sampled, y_sampled, selected_idx