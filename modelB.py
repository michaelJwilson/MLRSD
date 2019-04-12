def prep_model_2D_B(nhot=1, optimizer=adam, regress=True, loss='mae', nmesh=128):
  print("\n\nPreparing model.\n\n")

  _input = (nmesh, nmesh, 1)

  model  = Sequential()
  model.add(SeparableConv2D(32, (3, 3), activation='linear', input_shape=(128, 128, 1)))
  model.add(LeakyReLU(alpha=0.03))
  model.add(MaxPooling2D((2, 2)))
  model.add(SeparableConv2D(64, (3, 3), activation='linear'))
  model.add(LeakyReLU(alpha=0.03))
  model.add(MaxPooling2D((2, 2)))
  model.add(SeparableConv2D(64, (3, 3), activation='linear'))
  model.add(LeakyReLU(alpha=0.03))
  model.add(BatchNormalization())   ##  Batch renormalization should come late.                                                                                                                                     
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.1))

  if regress:
    ##  --  Appropriate [0, 1] output --                                                                                                                                                                            
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  else:
    model.add(Dense(nhot, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  model.summary()

  return  model
