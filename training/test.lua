-- Copyright 2015-2016 Carnegie Mellon University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.



testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testOnTrainLogger = optim.Logger(paths.concat(opt.save, 'test_on_train.log'))

local batchNumber
local triplet_loss
local timer = torch.Timer()

function test()
   print('[TESTING]')
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)
   
   p_test(testLoader, testLogger)
   
   print('==> doing epoch on train (not hard examples) data:')
   print("==> online epoch # " .. epoch)

   p_test(trainLoader, testOnTrainLogger)
   
end

function p_test(dataLoader, logger)

   batchNumber = 0
   if opt.cuda then
      cutorch.synchronize()
   end
   timer:reset()

   --model:evaluate()
   if opt.cuda then
      model:cuda()
   end

   triplet_loss = 0
   for i=1,opt.testEpochSize do
      donkeys:addjob(
         function()
            local inputs, scalarLabels, labels = dataLoader:sampleTriplet(opt.testBatchSize) --trainLoader:samplePeople(3, opt.testBatchSize)

            inputs = inputs:float()
            return sendTensor(inputs)
         end,
         testBatch
      )
      if i % 5 == 0 then
         donkeys:synchronize()
         collectgarbage()
      end
   end

   donkeys:synchronize()
   if opt.cuda then
      cutorch.synchronize()
   end

   triplet_loss = triplet_loss / opt.testEpochSize
   logger:add{
      ['avg triplet loss'] = triplet_loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average triplet loss (per batch): %.2f',
                       epoch, timer:time().real, triplet_loss))
   print('\n')


end

local inputsCPU = torch.FloatTensor()
local inputs
if opt.cuda then
   inputs = torch.CudaTensor()
else
   inputs = torch.FloatTensor()
end

function testBatch(inputsThread)
   receiveTensor(inputsThread, inputsCPU)
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   
   local embeddings = model:forward(inputs):float()
   local err = criterion:forward({
        embeddings:sub(1,opt.testBatchSize),
        embeddings:sub(opt.testBatchSize+1, 2*opt.testBatchSize),
        embeddings:sub(2*opt.testBatchSize+1, 3*opt.testBatchSize)})
   if opt.cuda then
      cutorch.synchronize()
   end

   triplet_loss = triplet_loss + err
   print(('Epoch: Testing [%d][%d/%d] Triplet Loss: %.2f'):format(epoch, batchNumber,
                                                                  opt.testEpochSize, err))
   batchNumber = batchNumber + 1
end
