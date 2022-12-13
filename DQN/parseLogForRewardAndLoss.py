import re
import sys

type = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] == "train" else "test"
file = "trained_models/dqn_model.log" if type == "train" else "test_dqn_model.log"

sampleRate = int(sys.argv[2]) if len(sys.argv) > 2 and int(sys.argv[2]) else 500
doSample = sampleRate > 1
avgRewardSample = 0
avgLossSample = 0

f = open(file, "r")
rewardCSV = open(f"all_ep_{type}_dqn_model_rewards.csv", "w")
lossCSV = open(f"all_ep_{type}_dqn_model_loss.csv", "w")
rewardCSV.write("Episode,Reward, Avg Reward\n")
lossCSV.write("Episode,Avg Step Loss, Overall Avg Loss\n")


if doSample:
    sampleRewardCSV = open(f"sample_ep_{type}_dqn_model_rewards.csv", "w")
    sampleLossCSV = open(f"sample_ep_{type}_dqn_model_loss.csv", "w")
    sampleRewardCSV.write(f"Sample ({sampleRate} Ep),Avg Reward ({sampleRate} Ep), Overall Avg Reward\n")
    sampleLossCSV.write(f"Sample ({sampleRate} Ep),Avg Loss ({sampleRate} Eps), Overall Avg Loss\n")

maxReward = 0
maxEp = 0
count = 0
totalReward = 0
totalLoss = 0
maxSavedStateReward = 0
maxSavedStateEpisode = 0
for line in f:
    count += 1
    episode = re.findall(r"Episode:\s+(-*\d+\.*\d*)\s+\|*", line)
    loss = re.findall(r"Loss:\s+(nan|-?\d+\.?\d*)\s+\|*", line)
    rewards = re.findall(r"Reward:\s+(-*\d+\.*\d*)\s+\|*", line)

    numEp = int(episode[0]) if episode else -1

    if rewards and episode:
        numReward = float(rewards[0])
        totalReward += numReward
        avgRewardSample += numReward

        if numReward > maxReward:
            maxReward = numReward
            maxEp = numEp

        if (numEp % 500) == 0 and numReward > maxSavedStateReward:
            maxSavedStateReward = numReward
            maxSavedStateEpisode = numEp

        if doSample and (numEp % sampleRate) == 0:
            avgSample = avgRewardSample / sampleRate if numEp != 0 else rewards[0]
            episodeMask = numEp / sampleRate
            sampleRewardCSV.write(f"{episodeMask},{avgSample},{totalReward/count}\n")
            avgRewardSample = 0

        rewardCSV.write(f"{episode[0]},{rewards[0]},{totalReward/count}\n")

    if loss and episode:
        numLoss = 0 if loss[0] == "nan" else float(loss[0])
        totalLoss += numLoss
        avgLossSample += numLoss

        if doSample and (numEp % sampleRate) == 0:
            avgSample = avgLossSample / sampleRate if numEp != 0 else loss[0]
            episodeMask = numEp / sampleRate
            sampleLossCSV.write(f"{episodeMask},{avgSample},{totalLoss/count}\n")
            avgLossSample = 0

        lossCSV.write(f"{episode[0]},{loss[0]},{totalLoss/count}\n")


print(f"Episode {maxEp} had max reward of {maxReward}")
print(f"Saved state with highest reward {maxSavedStateReward} is {maxSavedStateEpisode}")

f.close()
rewardCSV.close()
lossCSV.close()

if doSample:
    sampleRewardCSV.close()
    sampleLossCSV.close()
