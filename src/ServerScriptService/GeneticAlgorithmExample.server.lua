local PhysicsService = game:GetService("PhysicsService")
local Package = game:GetService("ReplicatedStorage").NNLibrary
local FeedforwardNetwork = require(Package.NeuralNetwork.FeedforwardNetwork)
local ParamEvo = require(Package.GeneticAlgorithm.ParamEvo)

local clock = os.clock()
local collisionGroupName = "CarCollisionsDisabled"

-- Setup CollisionGroup for cars
PhysicsService:CreateCollisionGroup(collisionGroupName)
PhysicsService:CollisionGroupSetCollidable(collisionGroupName, collisionGroupName, false)

-- Function to activate scripts inside of car
local function setupCar(car)
	car.RemoteControl.Torque = 1000
	car.RemoteControl.Throttle = 5
	car.RemoteControl.TurnSpeed = 30
	for _, item in pairs(car:GetDescendants()) do
		if item:IsA("BasePart") or item:IsA("UnionOperation") then
			PhysicsService:SetPartCollisionGroup(item, collisionGroupName)
		end
	end

	-- Create ray visualizers
	local ray = Instance.new("Part")
	ray.CanCollide = false
	ray.Massless = true
	ray.CFrame = car.RaycastPart.CFrame
	ray.Size = Vector3.new(1, 1, 1)
	ray.Color = Color3.fromRGB(255, 0, 0)
	ray.Name = "frontRay"
	ray.Parent = car
	local weld = Instance.new("WeldConstraint")
	weld.Part0 = car.RaycastPart
	weld.Part1 = ray
	weld.Parent = ray
end

-- Setup car
local carSource = game:GetService("ServerStorage").Car
setupCar(carSource)

-- Function that casts rays from the car in five directions and returns the distances in a table
local function getRayDistances(car)
	-- Setup RayCastParams
	local rayCastParams = RaycastParams.new()
	rayCastParams.IgnoreWater = true
	rayCastParams.FilterType = Enum.RaycastFilterType.Whitelist
	rayCastParams.FilterDescendantsInstances = {workspace.Walls}

	local bumper = car.RaycastPart
	local bumperPos = bumper.Position
	local dist = 250

	local FrontRay = workspace:Raycast(bumperPos, bumper.CFrame.RightVector * dist, rayCastParams)														-- Cast ray for front
	local FrontLeftRay = workspace:Raycast(bumperPos, Vector3.new(bumper.CFrame.RightVector.X, 0, bumper.CFrame.LookVector.Z) * dist, rayCastParams)	-- Cast ray to frontLeft
	local FrontRightRay = workspace:Raycast(bumperPos, Vector3.new(bumper.CFrame.RightVector.X, 0, -bumper.CFrame.LookVector.Z) * dist, rayCastParams)	-- Cast ray to frontRight
	local LeftRay = workspace:Raycast(bumperPos, bumper.CFrame.LookVector * dist, rayCastParams)														-- Cast ray to left
	local RightRay = workspace:Raycast(bumperPos, -bumper.CFrame.LookVector * dist, rayCastParams)														-- Cast ray to right

	local pos1 = 1
	local pos2 = 1
	local pos3 = 1
	local pos4 = 1
	local pos5 = 1
	if FrontRay then
		pos1 = ((FrontRay.Position - bumperPos).Magnitude) / dist
	end
	if FrontLeftRay then
		pos2 = ((FrontLeftRay.Position - bumperPos).Magnitude) / dist
	end
	if FrontRightRay then
		pos3 = ((FrontRightRay.Position - bumperPos).Magnitude) / dist
	end
	if LeftRay then
		pos4 = ((LeftRay.Position - bumperPos).Magnitude) / dist
	end
	if RightRay then
		pos5 = ((RightRay.Position - bumperPos).Magnitude) / dist
	end

	--local calc = pos1*dist
	--local ray = car.frontRay
	--ray.Size = Vector3.new(calc, 1, 1)
	--ray.Position = Vector3.new(bumperPos.X + calc/2, ray.Position.Y, ray.Position.Z)
	return {front = pos1, frontLeft = pos2, frontRight = pos3, left = pos4, right = pos5}
end

-- Settings for genetic algorithm
local geneticSetting = {
	--[[ The function that, when given the network, will return it's score.
	ScoreFunction = function(net)
		local startTime = os.clock()
		local clone = game:GetService("ServerStorage").Car:Clone()
		clone.RemoteControl.MaxSpeed = 200

		-- Parent to workspace and then setup Scripts of car
		clone.Parent = workspace

		local score = 0
		local bool = true
		local checkpointsHit = {}
		for _, v in pairs(clone:GetDescendants()) do
			if v:IsA("BasePart") and v.CanCollide == true then
				v.Touched:Connect(function(hit)
					if hit.Parent.Parent == workspace.Walls then	-- Destroy car on hit of walls
						bool = false
					elseif hit.Parent == workspace.Checkpoints and not checkpointsHit[tonumber(hit.Name)] then	-- Give extra points when car reaches checkpoint
						local numHit = tonumber(hit.Name)
						score += (numHit * 2)
						checkpointsHit[numHit] = hit
					end
				end)
			end
		end
		while bool do
			local distances = getRayDistances(clone)		-- Get Distances of rays
			local output = net(distances)					-- Get output of NN with input distances

			-- Set steering direction to direction of NN
			clone.RemoteControl.SteerFloat = output.steerDirection
			-- Set speed of car
			--clone.RemoteControl.MaxSpeed = math.abs(output.speed) * 300

			-- Check if this simulation has been running for longer than x seconds
			if os.clock() > startTime + 90 then
				score -= 40	-- Punish algorithm
				break
			end
			wait()
		end
		
		score += (os.clock() - startTime)/2		-- Increment score based on time alive (longer is better)
		print("Exit score: "..math.floor(score*100)/100)

		clone:Destroy()
		return score
	end;]]
	-- The function that runs when a generation is complete. It is given the genetic algorithm as input.
	PostFunction = function(geneticAlgo)
		local info = geneticAlgo:GetInfo()
		print("Generation "..info.Generation..", Best Score: "..info.BestScore)
	end;

	HigherScoreBetter = true;
	
	PercentageToKill = 0.4;
	PercentageOfKilledToRandomlySpare = 0.1;
	PercentageOfBestParentToCrossover = 0.8;
	PercentageToMutate = 0.7;
	
	MutateBestNetwork = true;
	PercentageOfCrossedToMutate = 0.6;
	NumberOfNodesToMutate = 3;
	ParameterMutateRange = 3;
}

local feedForwardSettings = {
	HiddenActivationName = "ReLU";
	OutputActivationName = "Tanh";
	Bias = 0;
    LearningRate = 0.1;
    RandomizeWeights = true;
}

-- Create a new network with 5 inputs, 2 layers with 4 nodes each and 1 output "steerDirection"
local tempNet = FeedforwardNetwork.new({"front", "frontLeft", "frontRight", "left", "right"}, 2, 4, {"steerDirection"}, feedForwardSettings)
--local tempNet = FeedforwardNetwork.newFromSave(game.ServerStorage.NetworkSave.Value)

local populationSize = 10
local geneticAlgo = ParamEvo.new(tempNet, populationSize, geneticSetting)		-- Create ParamEvo with the tempNet template, population size and settings

local scoreTable = {}
local generations = 50	-- Number of generations to train network with
for _ = 0, generations do
	for index = 1, populationSize do
		spawn(function()
			local startTime = os.clock()
			local clone = game:GetService("ServerStorage").Car:Clone()
			clone.RemoteControl.MaxSpeed = 200

			-- Parent to workspace and then setup Scripts of car
			clone.Parent = workspace

			local score = 0
			local bool = true
			local checkpointsHit = {}
			for _, v in pairs(clone:GetDescendants()) do
				if v:IsA("BasePart") and v.CanCollide == true then
					v.Touched:Connect(function(hit)
						if hit.Parent.Parent == workspace.Walls then	-- Destroy car on hit of walls
							bool = false
						elseif hit.Parent == workspace.Checkpoints and not checkpointsHit[tonumber(hit.Name)] then	-- Give extra points when car reaches checkpoint
							local numHit = tonumber(hit.Name)
							score += (numHit * 2)
							checkpointsHit[numHit] = hit
						end
					end)
				end
			end
			while bool do
				local distances = getRayDistances(clone)		-- Get Distances of rays
				local output
				--if firstRun then
				local population = geneticAlgo:GetPopulation()
				local net = population[1].Network
				output = net(distances)				-- Get output of NN with input distances

				-- Set steering direction to direction of NN
				clone.RemoteControl.SteerFloat = output.steerDirection
				-- Set speed of car
				--clone.RemoteControl.MaxSpeed = math.abs(output.speed) * 300

				-- Check if this simulation has been running for longer than x seconds
				if os.clock() > startTime + 90 then
					score -= 40	-- Punish algorithm
					break
				end
				wait()
			end
			
			score += (os.clock() - startTime)/2		-- Increment score based on time alive (longer is better)
			--print("Exit score: "..math.floor(score*100)/100)

			clone:Destroy()
			scoreTable[index] = score
		end)
		wait(1)
	end
	-- Wait until generation finished
	repeat
		wait(1)
		print("scoretable: "..#scoreTable)
	until #scoreTable >= populationSize
	
	geneticAlgo:ProcessGeneration(scoreTable)
	scoreTable = {}
end

local save = geneticAlgo:GetBestNetwork():Save()
game.ServerStorage.NetworkSave.Value = save
print(save)

--[[ * Code for running network
for i = 1, 20 do
	local clone = game:GetService("ServerStorage").Car:Clone()
	clone.RemoteControl.MaxSpeed = 200
	clone.Parent = workspace

	local bool = true
	for _, v in pairs(clone:GetDescendants()) do
		if v:IsA("BasePart") and v.CanCollide == true then
			v.Touched:Connect(function(hit)
				if hit.Parent.Parent == workspace.Walls then	-- Destroy car on hit of walls
					bool = false
				end
			end)
		end
	end
	while bool do
		local distances = getRayDistances(clone)	-- Get Distances of rays
		local output = tempNet(distances)			-- Get output of NN with input distances

		-- Set steering direction to direction of NN
		clone.RemoteControl.SteerFloat = output.steerDirection
		wait()
	end
	clone:Destroy()
end]]