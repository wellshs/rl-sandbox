def get_epsilon_function(epsilon_start, epsilon_final, epsilon_decay):
    def epsilon_by_frame(frame_idx):
        if frame_idx > epsilon_decay:
            return epsilon_final
        else:
            return epsilon_start + (epsilon_final - epsilon_start) * (frame_idx / epsilon_decay)
    return epsilon_by_frame
