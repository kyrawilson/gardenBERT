PennController.ResetPrefix(null); // Shorten command names (keep this line here))

DebugOff()   // Uncomment this line only when you are 100% done designing your experiment

// First show instructions, then experiment trials, send results and show end screen
Sequence("counter", "consent", "instructions", "practice", randomize("experimental-trial"), SendResults(), "end")

SetCounter("counter", "inc", 1);

// This is run at the beginning of each trial
Header(
    // Declare a global Var element "ID" in which we will store the participant's ID
    newVar("ID").global()    
)
.log( "id" , getVar("ID") ) // Add the ID to all trials' results lines

// Instructions
newTrial("consent",
     // Automatically print all Text elements, centered
    defaultText.center().print()
    ,
    newText("Welcome! Please begin by completing the following demographic survey.")
    ,
    newText("\n")
    ,
    newText("<a href='https://nyu.qualtrics.com/jfe/form/SV_20mWz94CICJtQOy' target='_blank'>Survey</a>")
    ,
    newText("\n")
    ,
    newText("When you are finished, type in your ID below and then click on the Continue button.")
        .center()
    ,
    newTextInput("inputID", "")
        .center()
        .css("margin","1em")    // Add a 1em margin around this element
        .print()
    ,
    newButton("Continue")
        .center()
        .print()
        // Only validate a click on Start when inputID has been filled
        .wait( getTextInput("inputID").testNot.text("") )
    ,
    // Store the text from inputID into the Var element
    getVar("ID").set( getTextInput("inputID") )
)


// Instructions
newTrial("instructions",
     // Automatically print all Text elements, centered
    defaultText.center().print()
    ,
    newText("In this task, you will have to read few sentences and answer questions about them.")
    ,
    newText("\n")
    ,
    newText("Press Start when you are ready to begin the practice session.")
    ,
    newText("\n")
    ,
    newButton("Start")
        .center()
        .print()
        .wait()
        // Only validate a click on Start when inputID has been filled
)

newTrial("practice", 
    defaultText.center().print()
    ,
    newText("The man walked to the store to get bread and milk.")
    ,
    
    newText("\n")
    , 
    
    newText("Press the spacebar when you have finished reading the sentence.")
        .css("font-style", "italic")
    , 
    
    newKey("spacebarp", " ")
        .wait()
    ,
    
    clear()
    ,
    newTimer("wait", 100)
    .start()
    .wait()
    ,
    newText("What did the man want to buy?")
    ,
    newText("\n")
    ,
    newTextInput("answer_boxp", "")
        .center()
        .log()
        .lines(0)
        .size(400, 40)
        .print()
    ,
    newText("\n")
    ,
    newButton("sendp", "Send")
        .center()
        .print()
        .wait( getTextInput("answer_boxp").testNot.text(" ") )
    , 
    clear()
    ,
    newTimer("wait", 100)
    .start()
    .wait()
    ,
    newText("Great job! Now you're ready for the experiment.")
    ,
    newText("\n")
    ,
    newText("Remember to read each sentence carefully and answer as accurately as possible!")
    ,
    newText("\n")
    ,
    newButton("Start")
        .center()
        .print()
        .wait()
        // Only validate a click on Start when inputID has been filled
    )

Template("GardenBERT_PC Ibex.csv", row =>
    newTrial("experimental-trial",
        defaultText.center().print()
        ,
        newText("context", row.context)
        ,
        
        newKey("spacebar", " ")
            .wait()
        ,
        
        clear()
        ,
        newTimer("wait", 100)
        .start()
        .wait()
        ,
        newText("question", row.question)
        ,
        newText("\n")
        ,
        newTextInput("answer_box", "")
            .center()
            .log()
            .lines(0)
            .size(400, 40)
            .print()
        ,
        newText("\n")
        ,
        newButton("send", "Send")
            .center()
            .print()
            .wait( getTextInput("answer_box").testNot.text(""))
        ,
        newVar("answer").global().set( getTextInput("answer_box") )
    )
    
    .log("code", row.code)
    .log("context", row.context)
    .log("question", row.question)
    .log("group", row.Group)
    .log("answer", getVar("answer"))
)

// Final screen
newTrial("end",
    newText("Thank you for your participation!")
        .center()
        .print()
    ,
    // This link a placeholder: replace it with a URL provided by your participant-pooling platform
    newText("<p><a href='https://app.prolific.co/submissions/complete?cc=C9O6ZAQJ' target='_blank'>Click here to validate your submission</a></p>")
        .center()
        .print()
    ,
    // Trick: stay on this trial forever (until tab is closed)
    newButton().wait()
)        
//.size(400, 200)



.setOption("countsForProgressBar",false)