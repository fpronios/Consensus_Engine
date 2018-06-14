#from keras.models import Sequential
#from keras.layers import Dense
#import numpy
# fix random seed for reproducibility
#from keras.callbacks import ModelCheckpoint
"""
def keras_learn(X, Y):
    model = Sequential()
    model.add(Dense(46, input_dim=45, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    model.load_weights("weights-improvement-01-0.99.hdf5")
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # checkpoint
    # filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    # Fit the model
    # model.fit(X, Y, validation_split=0.33, epochs=1500, batch_size=5, callbacks=callbacks_list, verbose=0) #, validation_split=0.33

    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # calculate predictions
    predictions = model.predict(X)
    # print('Predictions')
    # print(predictions)
    # round predictions
    rounded = [round(x[0]) for x in predictions]

    # print(rounded)
    # print(sum(rounded)/len(rounded))

    perm = predictions[:, 0].argsort()
    predictions = predictions[perm, 0]
    mols = np_arr[perm, 0]

    dfa = pd.read_sql_query("SELECT * FROM CDK5_active", conn)
    # print(df)
    active_araray = dfa.as_matrix(columns=['Active'])

    for r, x in zip(predictions, mols):
        if x in active_araray:
            print('Prediction: ', r, '  Molecule: ', x, '  Position: ', len(mols) - np.where(mols == x)[0])
"""