# Script for creating training, validation and testing sets
import os, shutil, random
import math

def create_datasets(datapath, training_prop, validation_prop, test_prop):

    jazzmaster = os.listdir(datapath + '/jazzmaster')
    lespaul = os.listdir(datapath + '/lespaul')
    mustang = os.listdir(datapath + '/mustang')
    prs_se = os.listdir(datapath + '/prs_se')
    sg = os.listdir(datapath + '/sg')
    stratocaster = os.listdir(datapath + '/stratocaster')
    telecaster = os.listdir(datapath + '/telecaster')

    total_files = 0

    for folder in ['jazzmaster', 'lespaul', 'mustang', 'prs_se', 'sg', 'stratocaster', 'telecaster']:
        total_files += len(eval(folder))
        print('{}: {}'.format(folder,len(eval(folder)) ))

    print('All files combined: {}'.format(total_files))

    train_total = 0
    validation_total = 0
    test_total = 0

    for folder in ['jazzmaster', 'lespaul', 'mustang', 'prs_se', 'sg', 'stratocaster', 'telecaster']:
        
        # Random shuffle the file list to make it random
        random.shuffle(eval(folder))
        
        # Determine the indexes
        training_size = math.floor(training_prop*len(eval(folder)))
        validation_size = math.floor(validation_prop*len(eval(folder)))
        testing_size = math.floor(test_prop*len(eval(folder)))

        # Take the samples
        train = eval(folder)[0:training_size]
        validation = eval(folder)[training_size:training_size+validation_size]
        test = eval(folder)[training_size+validation_size:]

        # Check for overlap
        assert [value for value in train if value in validation] == [] 
        assert [value for value in validation if value in test] == []
        assert [value for value in train if value in test] == []

        # Count the totals
        train_total += len(train)
        validation_total += len(validation)
        test_total += len(test)

        print('{} train: {}, validation: {}, test: {}'.format(folder, len(train), len(validation), len(test)))

        #Move the files
        
        for fname in train:
            srcpath = os.path.join(datapath + '/' + folder, fname)
            shutil.copyfile(srcpath, datapath + '/train/' + folder + '/' + fname)

        for fname in validation:
            srcpath = os.path.join(datapath + '/' + folder, fname)
            shutil.copyfile(srcpath, datapath + '/validation/' + folder + '/' + fname)

        for fname in test:
            srcpath = os.path.join(datapath + '/' + folder, fname)
            shutil.copyfile(srcpath, datapath + '/test/' + folder + '/' + fname)

    print('Totals: train: {}, validation {}, test: {}'.format(train_total,validation_total,test_total))


