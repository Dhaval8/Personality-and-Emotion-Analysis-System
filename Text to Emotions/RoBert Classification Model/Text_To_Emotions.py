

from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

def analyze_emotions_roberta(text, chunk_size=450):
    """
    Analyze emotions using RoBERTa model for emotion classification.
    
    Args:
        text (str): Input text to analyze
        chunk_size (int): Size of text chunks to process
    
    Returns:
        dict: Dictionary of emotions and their scores
    """
    # Initialize RoBERTa-based emotion analyzer
    emotion_analyzer = pipeline("text-classification", 
                              model="SamLowe/roberta-base-go_emotions", 
                              top_k=None)
    
    # Split text into manageable chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Analyze each chunk
    all_scores = []
    for chunk in chunks:
        try:
            result = emotion_analyzer(chunk)
            if isinstance(result, list) and all(isinstance(res, dict) for res in result[0]):
                scores_dict = {res['label']: res['score'] for res in result[0]}
                all_scores.append(scores_dict)
            else:
                print(f"Unexpected result format: {result}")
                continue
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            continue
    
    # Calculate average scores across all chunks
    final_scores = {}
    for score_dict in all_scores:
        for emotion, score in score_dict.items():
            if emotion not in final_scores:
                final_scores[emotion] = []
            final_scores[emotion].append(score)
    
    # Calculate mean scores
    average_scores = {emotion: np.mean(scores) for emotion, scores in final_scores.items()}
    
    # Sort emotions by score for better visualization
    return dict(sorted(average_scores.items(), key=lambda x: x[1], reverse=True))

def plot_roberta_emotions(emotion_scores, top_n=10):
    """
    Plot the top N emotions detected by RoBERTa.
    
    Args:
        emotion_scores (dict): Dictionary of emotions and their scores
        top_n (int): Number of top emotions to display
    """
    # Get top N emotions
    top_emotions = dict(list(emotion_scores.items())[:top_n])
    emotions = list(top_emotions.keys())
    scores = list(top_emotions.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(emotions, scores, color='lightcoral')
    plt.xlabel('Emotions')
    plt.ylabel('Scores')
    plt.title('Top Emotions Detected by RoBERTa Model')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Sample text for analysis
    text = """
    Okay, so I'm Dhaval Mehta, and let's begin the interview.
    So the first section planning and organizing.
    Tell me about a time when you had to manage multiple tasks or projects simultaneously, and how did you prioritize and organize your work to meet the deadlines? 
    Okay, so during my third year, I was managing an internship at dry Chem, working on a research paper on reinforcement learning and the stock market, forecasting and organizing adapt as a part of my IT committee, I prioritized tasks using a calendar system and allocated dedicated time slots for each of the four things. So for instance, I worked on a I worked on a coding SAP workflows on mornings, and conducted research meetings with my other research scholars in the afternoon and the evening slots were mainly dedicated for my IT committee work, and this structured approach allowed me to meet all of the deadlines successfully describe a situation where you Were responsible for planning the development of a complex app. 
    What steps did you take to ensure that the project was well organized and executed on schedule?
    Okay, so while designing an Android app for detecting internet connectivity and low battery issues, I begin by outlining the requirements creating wire frames and breaking the project into milestones like UI design, broadcast receiver integration and testing. Weekly reviews were conducted to ensure the process and this organized approach ensured the app was completed within the timeline and met all the functionality requirements as I had to submit this particular app as a practical in the next week. So it was a tedious task and a last project. So I streamlined each and every and I broke each and every large task into smaller tasks and worked on them simultaneously, together and then the question is.
    Can you give an example of a project where you had to adjust your initial plan due to unexpected challenges? How did you adapt to your approach while maintaining focus on the overall goals? 
    Okay, so during my internship at dry Chem, I encountered delays in automating the QC testing workflow due to unforeseen complexities in the data mapping and the data mining work. So, I reallocated time from other phases, like from my committee work and from my research work to debug the issue, and I also collaborated with my other team members to simplify the data mapping work and streamlined and with all of these things, I also streamlined testing to stay on track. This adjustment ensured the project met all of its overall goals without compromising the quality of the project.
    Toolshen coming to phase two, the communication skills
    Describe a time when you had to explain a technical issue or concept to a non technical team member or client. How did you ensure they understood your explanation? 
    Okay, so at dry Chem, I explained the automated workflow of the purchasing team to the purchasing team, and to ensure clarity, I used simple language and relatable algorithms and relatable materials, like pictures and flow charts, And comparing the process to a postal service. I also added some visual aids, like videos and other flow charts so that they can easily get to know the the small, small concepts used in the project, which helped the team quickly grasp the concept of courses and then tell me about a situation where clear communication was crucial to the success of an app development project, and how did you ensure all stakeholders were on the same page throughout the so while working on an Android app, I maintained clear communication by holding weekly meetings which stakeholders to share progress, gather feedback and address concerns. I also use detailed documentations and regular status updates to ensure everyone was aligned and everyone was knowing that what exactly is going on in the project.
    Can you share an example of a time when you had to present your app development progress to a senior manager or client, and how did you tailor your message to meet the level of understanding and explanations, exceptions, expectations? 
    So during my internship, I presented the automated QC workflow to the senior management um under whom I was mainly doing the project, and I started with an overview of the problem and its impact. I explained the solution through visual aids and highlighted key results like time savings and improved accuracy. This approach made the presentation impactful, and it also made it easy to flow and follow.
    Then comes the phase three, the interpersonal skills.
    Tell me about a time when you worked closely with a colleague who had a different perspective of or work style than you. How did you manage to collaborate effectively? 
    There are many different times where I worked with I had actually worked with many different colleagues. So also at the time of my internship, I worked with a team of four to five members and who preferred a different database, data structure for the QC sample table. We discussed our approaches, evaluated their pros, cons, and my suggestions were also given and combined the best aspects for both the ideas, the collaboration improved our solution and strengthened our work relationship. Describe a situation where you had to handle conflict or disagreement with a team member during an app development project, how did you address the issue to maintain a positive working relationship. Okay, so this was a time while I was working for the IIT committee, so we were planning the ULECTRO event, there was a scheduling disagreement among the team members, and I facilitated a discussion to understand everyone's concerns and everyone's point of view, and also proposed a compromise, so that everybody get to know that what are the advantages and disadvantages of their ideas, and why should we not take their ideas, but we should take the other people's ideas, other person's ideas, and ensured alignment with the Event goals that ensured alignment with the event goals, yes, and this resolution maintained a positive environment and ensured the events success at the end.
    Can you share an example of how you build rapport, report and trust with a cross functional team, example, designers, product managers, etc. And how did that collaboration led to the success of the project?
    At dry Chem I worked with both technical and non-technical teams, and by being approachable, transparent and proactive in designing their concerns, in addressing their concerns, I build trust amongst them, and this collaboration ensured a smooth implementation of automated workflow at the end of the project,
        section four is the teamwork section
    Tell me about a time when you worked as a part of a development team under tight deadlines, and how did you collaborate to ensure that the project was delivered on time.
    So again, at Drychem, our team faced tight deadlines to complete the QC testing automation. We divided the tasks, supported each other during the bottlenecks and maintained daily progress updates. This teamwork ensured timely delivery and the perfect success of our QC project.
    During an instance when you when your team faced a technical challenge, how did you contribute to solving the problem, and how did the team work together to overcome it?
    So, okay, so during my stock market forecast research, we face challenges with the volatile data, like the data was continuously changing and we were not able to get a perfect, precise or concise answer, so I suggested assembling learning and symbol. Learning to improve model robustness, and the team collaborated to implement and test this approach, resulting in better predictions. So at the end of the day, we finally got the results, and we made that particular research people successfully. 
    Can you provide an example of when you had to rely on others in your team to complete a project, and how did you ensure that the team, collective teams, collective effort, led to successful outcomes?
    Okay, so this is a time when we were working on the reinforcement learning project, and I relied on my teammates for literature reviews and model comparisons. So by coordinating tasks and encouraging regular updates, we ensured a cohesive and impactful paper, and luckily, that paper also got accepted in the ICPC2T 2025, international conference held by IEEE in the NIT riper college.
    Section, five is the attention to detail.
    Tell me about a time when you identified a small but critical bug or issue in your code that others missed. How did you spot it and what happened? What steps did you take to fix it?
    Okay, so again, in dry cam, I noticed a small mismatch in the data mapping for QC results, for the QC test results that others missed, and I carefully reviewed the logic, identified the error and fixed it. It took me a couple of attempts to make it perfect, but ensuring the systems generated accurate reports, and I also held on to that and made the project work. 
    At the end of the day, describe a situation where you had to ensure your code met all requirements and was bug free before deployment. How did you ensure attention to detail throughout the development process? 
    Okay, so while, uh, developing my recipe app, as I said before, we had tight guidelines first of all, and just two days before the deployment, and just two days before the submission, we had a small meet where we all came together and we reviewed the whole project, line by line, and yes, we found a few bugs, and at the same at the same point of time, we also collected those bugs and made the code work smoothly flawlessly.
    Can you share an example of time when your attention to detail improved the user experience of an app or resolved a performance issue? 
    So while testing the Android app, I noticed a delay in connectivity detection. So for example, we had to collect our Android app with the Firebase. So first of all, it needs to take data from the Firebase and update it itself dynamically so that the user can get to know the login information and the selected recipes. So first of all, it wasn't there was some lag or there was some issue that was stopping it from happening. But by optimizing the broadcast receiver logic, I improved the response time, enhancing the user experience at the end.
    Section six, creativity and innovation. 
    Tell me about a time when you had to come up with a creative solution to a technical problem during app development. What was the problem and how did you? How did your creative thinking help solve it? 
    Okay, so during my stock market forecast research handling volatile data was a big challenge, as I said earlier, and I proposed combining predictions from multiple models using n symbol learning, and which significantly improved the accuracy, as we have used around four to six models in total. And hence, at the end, we came up with a final solution, final output. 
    Describe a situation where you implemented an innovative feature or functionality that led the user experience or the app's performance, that enhanced the user experience of the apps performance, and what was your thought process behind the innovation? 
    So for my Android app, I designed real time modifications for low battery and connectivity issues, and using intuitive UI elements, this innovation improved usability and User enhancement and user engagement so that the user won't be won't need to reach out. Charge or would need to charge their device again and again, as the battery usage of our app would be minimum. Can you provide an example of a project where you had to think outside the box to overcome technical constraints or limitations, and how did your creative approach led to successful outcomes. So while designing the QC database at dry Chem, I restructured the table to simplify the queries and improved report generation speed, which optimized the entire workflow, and it also smoothened the overall workflow. So yeah. 
    Okay, so section number seven, result orientation. 
    Describe a project where you were under pressure to deliver an app or feature on time, and how did you stay focused on the end result and ensure it was delivered within the deadline.
    So during my internship at dry camp, again, I was tasked with automating the QC testing workflow within a strict timeline. And to stay focused, I broke down the project into smaller tracks, like designing the database, creating SAP screens and implementing the email automation process, and also maintained a checklist, prioritizing each task based on their impact in the final outcome and regular updates to my manager ensured we stayed aligned, and at the end, the project was successfully delivered On time, streamlining the testing process for the company. 
    Tell me about a time when you took the initiative to go above and beyond in an app development project to achieve key outcomes, and What actions did you take and what was the result? 
    So while I was working on my recipe app, during my mobile application development project, my app had many connectivity issues, so I noticed that the basic functionality was sufficient but lacked user engagement. So I took the initiative to design and integrate a user friendly notification system using toast messages and intuitive graphics, and this additional feature improved the satisfaction and the app apart in terms of usability. 
    Can you give an example of a situation where you were able to achieve significant improvements or results for the product by making strategic decisions or adjustments in your development approach.
    So in my stock market forecasting research, I initially faced challenges with the model accuracy due to the changing data, and the due to the changing time frames which in which we were using the data, and I strategically, decided to shift from a single LSTM model to an ensemble approach, where we combine different multiple models to ensure prediction reliability. And this decision significantly improved the results, leading to a more robust forecastingframework.
    Session number eight, problem solving and decision making.
    Tell me about a time when you encountered a major technical challenge during the app development, and how did you how did you approach the problem, and what decisions make, making process did you use to find a solution? 
    Okay, so again, during my internship at dry camp, I faced a challenge in automating the QC test result generation due to the data mismatches, as some of the data were in teacher oriented and somewhere um character oriented. So to solve this, I firstly analyzed the workflow to identify the root cause, and then collaborated with the QC team to understand the data requirements, and then I adjusted the database structure as per the requirements and optimized the app logic to resolve the issue, and this systematic approach ensured accurate and efficient report generation.
    Describe a situation where you had to make a decision between different technical approaches to implement a feature, and how did you weigh the pros and cons of Each option and what factors influence your final decision? 
    So while developing my Android app, I the mobile application development project, I had to choose between using broadcast receivers or services for real time collectivity detection as both were. Having its own pros and cons, and like the broadcast receivers were lighter and faster and event based triggers, and they were faster for the event based triggers and services offered continuous monitoring, but were resource intensive, like they consume a lot of battery and a lot of data, and this and by considering these points, the app scope and the user experience, I chose the broadcast receivers which provided the needed functionality and in a very efficient manner. 
    Can you share an example of the time when you identified a potential issue early in the development cycle and took proactive steps to address it before it became a larger problem. Okay, so during my internship again and in the tricam, I noticed early on that the QC sample database design lacked flexibility for future scalability. So to prevent potential bottlenecks, I proposed and implemented a more modular design, like I broke, I just broke down the particular project, new different modules, and that accommodated the additional testing parameters and streamlined data entry, so this proactive step avoided significant work later, and ensured the system longitivity.
    Thank you so much for the interview.
    
    """
    
    # Analyze emotions
    print("Analyzing emotions using RoBERTa model...")
    emotion_scores = analyze_emotions_roberta(text)
    
    # Print detailed scores
    print("\nDetailed Emotion Scores:")
    for emotion, score in emotion_scores.items():
        print(f"{emotion}: {score:.4f}")
    
    # Plot results
    plot_roberta_emotions(emotion_scores)


