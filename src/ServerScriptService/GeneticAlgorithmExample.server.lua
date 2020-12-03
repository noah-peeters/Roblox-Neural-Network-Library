local HttpService = game:GetService("HttpService")
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
	car.RemoteControl.MaxSpeed = 200
	car.RemoteControl.Torque = 40
	car.RemoteControl.Throttle = 1
	car.RemoteControl.TurnSpeed = 25
	for _, item in pairs(car:GetDescendants()) do
		if item:IsA("BasePart") then
			PhysicsService:SetPartCollisionGroup(item, collisionGroupName)
		end
	end

	-- Parent to workspace and then setup Scripts of car
	car.Parent = workspace
	for _, item in pairs(car:GetDescendants()) do
		if item:IsA("Script") then
			item.Disabled = false
		end
	end
end

-- Function that casts rays from the car in five directions and returns the distances in a table
local function getRayDistances(car)
	-- Setup filterTable for rays to ignore (all parts of car)
	local filterTable = {}
	for _, v in pairs(car:GetDescendants()) do
		if v:IsA("BasePart") then
			table.insert(filterTable, v)
		end
	end
	-- Setup RayCastParams
	local rayCastParams = RaycastParams.new()
	rayCastParams.IgnoreWater = true
	rayCastParams.FilterType = Enum.RaycastFilterType.Blacklist
	rayCastParams.FilterDescendantsInstances = {car}

	local distance = 50

	local bumper = car.MainBumper
	local bumperPos = bumper.Position
	local bumperCFrame = bumper.CFrame

	local function getDirection(vec)
		local start = bumperPos
		local finish = (bumperCFrame * CFrame.new(vec)).Position
		return (finish - start).Unit * distance
	end

	local FrontRay = workspace:Raycast(bumperPos, getDirection(Vector3.new(0,0,-1)), rayCastParams)
	local FrontLeftRay = workspace:Raycast(bumperPos, getDirection(Vector3.new(-1,0,-1)), rayCastParams)
	local FrontRightRay = workspace:Raycast(bumperPos, getDirection(Vector3.new(1,0,-1)), rayCastParams)
	local LeftRay = workspace:Raycast(bumperPos, getDirection(Vector3.new(-1,0,0)), rayCastParams)
	local RightRay = workspace:Raycast(bumperPos, getDirection(Vector3.new(1,0,0)), rayCastParams)

	local pos1 = 1
	local pos2 = 1
	local pos3 = 1
	local pos4 = 1
	local pos5 = 1
	if FrontRay then
		pos1 = ((FrontRay.Position - bumperPos).Magnitude) / distance
	end
	if FrontLeftRay then
		pos2 = ((FrontLeftRay.Position - bumperPos).Magnitude) / distance
	end
	if FrontRightRay then
		pos3 = ((FrontRightRay.Position - bumperPos).Magnitude) / distance
	end
	if LeftRay then
		pos4 = ((LeftRay.Position - bumperPos).Magnitude) / distance
	end
	if RightRay then
		pos5 = ((RightRay.Position - bumperPos).Magnitude) / distance
	end
	return {front = pos1, frontLeft = pos2, frontRight = pos3, left = pos4, right = pos5}
end

-- Settings for genetic algorithm
local geneticSetting = {
	--The function that, when given the network, will return it's score.
	ScoreFunction = function(net)
		local startTime = os.clock()
		local car = game:GetService("ServerStorage").Car:Clone()
		-- Setup car
		setupCar(car)

		local bool = true
		for _, v in pairs(car:GetDescendants()) do
			if v:IsA("BasePart") then
				v.Touched:Connect(function(hit)
					if hit.Parent.Parent == workspace.Obstacles then
						bool = false
					end
				end)
			end
		end
		while bool do
			local distances = getRayDistances(car)		-- Get Distances of rays
			local output = net(distances)				-- Get output of NN with input distances
			local steeringDir = output.steerDirection
			
			--car.RemoteControl.Steer = steeringDir
			car.LeftMotor.DesiredAngle = steeringDir
			car.RightMotor.DesiredAngle = steeringDir
			wait()
		end
		
		local score = os.clock() - startTime	-- Get score based on time alive (longer is better)
		print("Exit score: "..math.floor(score*100)/100)

		car:Destroy()
		return score
	end;
	--The function that runs when a generation is complete. It is given the genetic algorithm as input.
	PostFunction = function(geneticAlgo)
		local info = geneticAlgo:GetInfo()
		print("Generation "..info.Generation..", Best Score: "..info.BestScore)
	end;
	HigherScoreBetter = true;
	
	PercentageToKill = 0.6;
	PercentageOfKilledToRandomlySpare = 0.1;
	PercentageOfBestParentToCrossover = 0.6;
	PercentageToMutate = 0.3;
	
	MutateBestNetwork = false;
	PercentageOfCrossedToMutate = 0.5;
	NumberOfNodesToMutate = 2;
	ParameterMutateRange = 4;
}

local feedForwardSettings = {
	HiddenActivationName = "ReLU";
	OutputActivationName = "Tanh";
}

-- Create a new network with 5 inputs, 2 layers with 4 nodes each and 1 output "steerDirection"
local tempNet = FeedforwardNetwork.new({"front", "frontLeft", "frontRight", "left", "right"}, 2, 4, {"steerDirection"}, feedForwardSettings) --FeedforwardNetwork.newFromSave(game.ServerStorage.NetworkSave.Value)

-- Create ParamEvo with the tempNet template, population size and settings
local geneticAlgo = ParamEvo.new(tempNet, 20, geneticSetting)

-- Run the algorithm x generations1
geneticAlgo:ProcessGenerationsInBatch(100, 0.5, 1)

-- Get the best network in the population
local net = geneticAlgo:GetBestNetwork()
local save = net:Save()
local stringSave = HttpService:JSONEncode(save)

print(stringSave)
game.ServerScriptService.NetworkSave.Value = stringSave