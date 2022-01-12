class EpsilonScheduler:
    """ 
    EpsilonScheduler should help to schedule a value of epsilon 
    for exploration-exploitation based on the current iteration/frame.
    This scheduler does linear annealing between anneal_start and anneal_end.
    """

    def __init__(self, epsilon_start=1, epsilon_end=0.1, anneal_start=5e4, anneal_end=1e6):
        """
        :param epsilon_start: Initial epsilon
        :param epsilon_end: Final epsilon after annealing
        :param anneal_start: The first frame to start annealing
        :param anneal_end: The last frame to stop annealing
        """
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end

    def __call__(self, frame_num):
        """
        Returns an epsilon value given the current iteration/frame number.

        :param frame_num: The current frame_number
        """
        if frame_num <= self.anneal_start:
            return self.epsilon_start
        elif frame_num > self.anneal_end:
            return self.epsilon_end
        else:
            slope = (self.epsilon_start - self.epsilon_end)/(self.anneal_end - self.anneal_start)
            return self.epsilon_start - slope * (frame_num - self.anneal_start)

if __name__ == "__main__":
    scheduler = EpsilonScheduler()

    from matplotlib import pyplot as plt
    # Plot the epsilon scheduling for sanity checking
    eps = [scheduler(i) for i in range(1500000)]
    plt.plot(eps)
    plt.show()
