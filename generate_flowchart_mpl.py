import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_flowchart(output_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Define box properties
    box_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5)
    shadow_props = dict(boxstyle='round,pad=0.3', facecolor='lightgray', edgecolor='none')
    
    # Helper to draw box with shadow and connecting line
    def draw_box(text, x, y, width=2.5, height=1.0, color='white', next_box=None):
        # Shadow
        rect_shadow = patches.FancyBboxPatch((x - width/2 + 0.05, y - height/2 - 0.05), width, height, 
                                             boxstyle="round,pad=0.1", fc='lightgray', ec='none', zorder=1)
        ax.add_patch(rect_shadow)
        
        # Main box
        rect = patches.FancyBboxPatch((x - width/2, y - height/2), width, height, 
                                      boxstyle="round,pad=0.1", fc=color, ec='black', lw=1.5, zorder=2)
        ax.add_patch(rect)
        
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3, wrap=True)

        if next_box:
            # Draw arrow
            ax.annotate('', xy=(next_box[0], next_box[1] + next_box[3]/2 + 0.1), xytext=(x, y - height/2),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), zorder=1)

    # Column 1: Data Curation
    ax.text(2, 9.5, "1. Data Curation\n(CuradoriaQSAR)", ha='center', fontsize=12, fontweight='bold', color='#005a9c')
    draw_box("Raw Dataset\n(CSV / Excel)", 2, 8.5, color='#E0E0E0', next_box=(2, 7.0, 2.5, 1.0))
    draw_box("Chemical Normalization\n(Salt Removal)", 2, 7.0, next_box=(2, 5.5, 2.5, 1.0))
    draw_box("SMILES Canonicalization\n(RDKit)", 2, 5.5, next_box=(2, 4.0, 2.5, 1.0))
    draw_box("Activity Standardization\n(nM / pIC50)", 2, 4.0, next_box=(2, 2.5, 2.5, 1.0))
    draw_box("Duplicate Filtering\n(GeoMean / Discordance)", 2, 2.5, next_box=(2, 1.0, 2.5, 1.0))
    draw_box("Curated Dataset", 2, 1.0, color='#f2f2f2')

    # Connector to Col 2
    ax.annotate('', xy=(4.75, 8.5), xytext=(3.25, 1.0),
                arrowprops=dict(facecolor='gray', shrink=0.05, width=1.5, headwidth=8, connectionstyle="arc3,rad=-0.2"), zorder=0)

    # Column 2: QSAR Modeling
    ax.text(6, 9.5, "2. QSAR Modeling\n(ModeladorQSAR)", ha='center', fontsize=12, fontweight='bold', color='#005a9c')
    
    draw_box("Feature Engineering\n(Fingerprints)", 6, 8.5, next_box=(6, 7.0, 2.5, 1.0))
    draw_box("Data Splitting\n(Stratified Train/Test)", 6, 7.0, next_box=(6, 5.5, 2.5, 1.0))
    draw_box("Applicability Domain\n(k-NN, Z < 3.0)", 6, 5.5, next_box=(6, 4.0, 2.5, 1.0))
    draw_box("Model Training\n(RF, SVM, GBM, KNN)", 6, 4.0, next_box=(6, 2.5, 2.5, 1.0))
    draw_box("Validation & Ranking\n(MCC, AUC)", 6, 2.5, color='#e6f3ff')

    # Connector to Col 3
    ax.annotate('', xy=(8.75, 8.5), xytext=(7.25, 2.5),
                arrowprops=dict(facecolor='gray', shrink=0.05, width=1.5, headwidth=8, connectionstyle="arc3,rad=-0.2"), zorder=0)

    # Column 3: Virtual Screening
    ax.text(10, 9.5, "3. Virtual Screening\n& Deployment", ha='center', fontsize=12, fontweight='bold', color='#005a9c')
    
    draw_box("Select Best Model\n(Max MCC)", 10, 8.5, next_box=(10, 7.0, 2.5, 1.0))
    draw_box("Generate PDF Report\n(ROC Curves)", 10, 7.0, color='#f2f2f2', next_box=(10, 5.5, 2.5, 1.0))
    draw_box("Virtual Screening\n(New Compounds)", 10, 5.5, next_box=(10, 4.0, 2.5, 1.0))
    draw_box("Prediction Results\n(Probabilities)", 10, 4.0, color='#E0E0E0')

    # Separator Lines
    ax.plot([4, 4], [0.5, 9.8], color='gray', linestyle='--', linewidth=0.8)
    ax.plot([8, 8], [0.5, 9.8], color='gray', linestyle='--', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Workflow diagram saved to: {output_path}")

if __name__ == "__main__":
    create_flowchart('docs/assets/screensar_workflow.png')
